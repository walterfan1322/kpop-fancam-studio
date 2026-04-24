// Typed API client for the FastAPI backend. Base URL is proxied by Vite in
// dev; in prod set VITE_API_BASE.
const BASE = import.meta.env.VITE_API_BASE || ''

export type GroupSummary = {
  name: string
  artist_name: string
  mb_artist_id: string | null
  track_count: number
  mp3_count: number
}

export type TrackOut = {
  title: string
  music_url: string | null
  has_mp3: boolean
  mp3_size_kb: number | null
  status: string | null
}

export type GroupDetail = GroupSummary & { tracks: TrackOut[] }

export type JobOut = {
  id: string
  kind: string
  status: 'pending' | 'running' | 'done' | 'failed'
  return_code: number | null
  started_at: number
  finished_at: number
  lines_count: number
  // Optional: older job records predate these fields, so treat as optional
  // on the frontend even though the backend populates them for new jobs.
  script?: string
  args?: string[]
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status} ${text}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listGroups: () => call<GroupSummary[]>('/api/groups'),
  getGroup: (name: string) => call<GroupDetail>(`/api/groups/${encodeURIComponent(name)}`),
  getMembers: (name: string, refresh = false) =>
    call<{ latin: string; hangul: string; chinese: string }[]>(
      `/api/groups/${encodeURIComponent(name)}/members${refresh ? '?refresh=true' : ''}`,
    ),
  replaceTracks: (name: string, tracks: { title: string; music_url: string | null }[]) =>
    call<{ ok: boolean; count: number }>(
      `/api/groups/${encodeURIComponent(name)}/tracks`,
      { method: 'PUT', body: JSON.stringify({ tracks }) },
    ),
  deleteGroup: (name: string) =>
    call<{ ok: boolean }>(
      `/api/groups/${encodeURIComponent(name)}`,
      { method: 'DELETE' },
    ),

  startFetch: (group: string, limit = 40) =>
    call<JobOut>('/api/jobs/fetch', {
      method: 'POST',
      body: JSON.stringify({ group, limit }),
    }),
  startDownload: (group: string | null, delay = 3, headed = false) =>
    call<JobOut>('/api/jobs/download', {
      method: 'POST',
      body: JSON.stringify({ group, delay, headed }),
    }),
  startResolve: (group: string, urls: string[], threshold = 0.85) =>
    call<JobOut>('/api/jobs/resolve', {
      method: 'POST',
      body: JSON.stringify({ group, urls, threshold, dry_run: false }),
    }),
  getJob: (id: string) => call<JobOut>(`/api/jobs/${id}`),
  listJobs: () => call<JobOut[]>('/api/jobs'),
  getJobLogs: (id: string, since: number) =>
    call<{ status: string; return_code: number | null; next_offset: number; lines: string[] }>(
      `/api/jobs/${id}/logs?since=${since}`,
    ),

  audioUrl: (group: string, title: string) =>
    `${BASE}/api/audio/${encodeURIComponent(group)}/${encodeURIComponent(title)}.mp3`,

  // Videos & clips
  listVideos: () => call<VideoMeta[]>('/api/videos'),
  downloadVideo: (url: string) =>
    call<JobOut>('/api/videos/download', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),
  deleteVideo: (stem: string) =>
    call<{ ok: boolean }>(
      `/api/videos/${encodeURIComponent(stem)}`,
      { method: 'DELETE' },
    ),
  videoUrl: (stem: string) =>
    `${BASE}/api/videos/${encodeURIComponent(stem)}/file`,

  listClips: () => call<ClipOut[]>('/api/videos/clips'),
  clipUrl: (group: string, song: string, title: string) =>
    `${BASE}/api/videos/clips/${encodeURIComponent(group)}/${encodeURIComponent(song)}/${encodeURIComponent(title)}`,
  // Download URL with ?keep=1 — server touches a `.keep` sidecar so the clip
  // won't be swept on the next oneshot run.
  clipDownloadUrl: (group: string, song: string, title: string) =>
    `${BASE}/api/videos/clips/${encodeURIComponent(group)}/${encodeURIComponent(song)}/${encodeURIComponent(title)}?keep=1`,
  markClipKept: (group: string, song: string, title: string) =>
    call<{ ok: boolean }>(
      `/api/videos/clips/${encodeURIComponent(group)}/${encodeURIComponent(song)}/${encodeURIComponent(title)}/keep`,
      { method: 'POST' },
    ),
  // 'Hide from Recent Clips' — writes a `.hidden` sidecar on the server
  // so list_clips omits it thereafter. The mp4 itself is not touched
  // (normal sweep/keep rules still apply). Lets the user permanently
  // dismiss a clip from the UI without deleting the file.
  markClipHidden: (group: string, song: string, title: string) =>
    call<{ ok: boolean }>(
      `/api/videos/clips/${encodeURIComponent(group)}/${encodeURIComponent(song)}/${encodeURIComponent(title)}/hide`,
      { method: 'POST' },
    ),
  deleteClip: (group: string, song: string, title: string) =>
    call<{ ok: boolean }>(
      `/api/videos/clips/${encodeURIComponent(group)}/${encodeURIComponent(song)}/${encodeURIComponent(title)}`,
      { method: 'DELETE' },
    ),

  startMatch: (group: string, video_stem: string, threshold = 0.6, margin = 0.03) =>
    call<JobOut>('/api/match', {
      method: 'POST',
      body: JSON.stringify({ group, video_stem, threshold, margin, extract: true }),
    }),

  searchVideos: (body: {
    queries: string[]; limit?: number;
    min_dur?: number; max_dur?: number;
    min_height?: number; min_views?: number;
    title_any?: string[];
  }) =>
    call<Candidate[]>('/api/videos/search', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  downloadBatch: (urls: string[], max_height = 1080) =>
    call<JobOut>('/api/videos/download-batch', {
      method: 'POST',
      body: JSON.stringify({ urls, max_height }),
    }),
  probeVideo: (video_stem: string) =>
    call<JobOut>('/api/videos/probe', {
      method: 'POST',
      body: JSON.stringify({ video_stem }),
    }),
  oneshot: (body: { group: string; song: string; member_lat: string; member_han?: string; count: number; delogo_corners?: ('tl' | 'tr' | 'bl' | 'br' | 'auto')[]; force_landscape?: boolean; merge_sources?: number; merge_style?: 'xfade' | 'hard_cut'; use_pose?: boolean }) =>
    call<JobOut>('/api/videos/oneshot', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

export type Quality = {
  cuts: number
  cuts_per_sec: number
  duration_sec: number
  median_persons: number
  median_target_h_ratio: number
  target_cx_std: number
  scores: {
    s_cuts: number
    s_persons: number
    s_height: number
    s_stability: number
    composite: number
  }
  elapsed_sec: number
}

export type VideoMeta = {
  id: string
  stem: string
  title: string
  url: string
  duration: number | null
  path: string
  size_mb: number | null
  has_file: boolean
  quality: Quality | null
}

export type Candidate = {
  id: string
  url: string
  title: string
  uploader: string | null
  channel_id: string | null
  duration: number | null
  view_count: number | null
  height: number | null
  passed: boolean
  reject_reasons: string[]
  already_downloaded: boolean
}

export type ClipOut = {
  group: string
  title: string
  song: string
  path: string
  size_mb: number | null
  // Older backends don't populate this — treat as optional on the client.
  mtime?: number
}
