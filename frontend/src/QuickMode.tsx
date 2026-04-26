import { useEffect, useRef, useState } from 'react'
import { api, type GroupDetail, type GroupSummary, type JobOut } from './api'
import { JobLog, type OneshotProgress } from './JobLog'
import { JobsHistory } from './JobsHistory'
import { RecentClips } from './RecentClips'
import { useT } from './i18n'

// localStorage key for persisting the currently-running oneshot job so the
// user can refresh / close-reopen the tab without losing progress.
const SAVED_JOB_KEY = 'quickmode.activeJob.v1'
type SavedJob = {
  jobId: string
  group: string
  song: string
  memberLat: string
  memberHan: string
  memberChi: string
  count: number
  autoDelogo: boolean
  forceLandscape?: boolean
  mergeSources?: number
  mergeStyle?: 'xfade' | 'hard_cut'
  usePose?: boolean
}

type Member = { latin: string; hangul: string; chinese: string }

function ProgressBar({ p, t }: { p: OneshotProgress; t: (k: any, v?: any) => string }) {
  // Base the bar on completed attempts out of pool size once we know the
  // pool; before that, fall back to matched/target so the user at least
  // sees movement during the search phase.
  const pct = p.pool > 0
    ? Math.min(100, Math.round((p.done / p.pool) * 100))
    : (p.target > 0 ? Math.min(100, Math.round((p.matched / p.target) * 100)) : 0)
  const label =
    p.phase === 'searching' ? t('progSearching') :
    p.phase === 'picking'   ? t('progPicking') :
    p.phase === 'working'   ? (p.current ?? t('progWorking')) :
                              t('progDone')
  const counters =
    p.pool > 0 ? `${p.done}/${p.pool}` : ''
  const matched = p.target > 0 ? `${p.matched}/${p.target}` : ''
  return (
    <div>
      <div className="progress-row">
        <span className="progress-label">{label}</span>
        {counters && <span className="progress-counter">{counters}</span>}
        {matched && (
          <span className="progress-counter progress-matched">
            {t('progMatched')} {matched}
          </span>
        )}
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function QuickMode() {
  const t = useT()
  const [groups, setGroups] = useState<GroupSummary[]>([])
  const [group, setGroup] = useState('')
  const [detail, setDetail] = useState<GroupDetail | null>(null)
  const [song, setSong] = useState('')
  const [members, setMembers] = useState<Member[]>([])
  const [memberIdx, setMemberIdx] = useState<number>(-1)
  const [memberLat, setMemberLat] = useState('')
  const [memberHan, setMemberHan] = useState('')
  const [memberChi, setMemberChi] = useState('')
  const [membersLoading, setMembersLoading] = useState(false)
  const [count, setCount] = useState(1)
  const [autoDelogo, setAutoDelogo] = useState(true)
  const [forceLandscape, setForceLandscape] = useState(false)
  // Multi-source merge: when enabled, we pull up to `mergeSources` matched
  // source videos and fuse them into ONE clip that hops between angles
  // second-by-second. Disabled by default because it's slower and costs
  // more bandwidth (N downloads + tracked-segment computes).
  const [mergeEnabled, setMergeEnabled] = useState(false)
  const [mergeSources, setMergeSources] = useState(3)
  // hard_cut is the "outfit-swap fancam" look — instant outfit/venue change
  // at each merge boundary. Default stays on xfade because hard-cut only
  // looks good once sources are same-angle same-framing (M3 canonical
  // framing). Ignored when merge is off.
  const [mergeStyle, setMergeStyle] = useState<'xfade' | 'hard_cut'>('xfade')
  // Pose-guided canonical framing (M3b) + angle-bucket cut preference
  // (M4b). Adds ~30s of CPU inference per source but produces much more
  // consistent head placement and prefers same-angle cuts over 3/4 rotations.
  // Off by default because it only matters for the outfit-swap look.
  const [usePose, setUsePose] = useState(false)
  const [job, setJob] = useState<JobOut | null>(null)
  const [progress, setProgress] = useState<OneshotProgress | null>(null)
  const [err, setErr] = useState<string | null>(null)
  // While a restore is in flight the group-change effect should NOT clobber
  // the song / member fields we've just rehydrated from localStorage.
  const restoringRef = useRef<SavedJob | null>(null)
  const [summary, setSummary] = useState<{
    group: string
    song: string
    matched: number
    considered: number
    clips: { video_id: string; video_stem: string; video_title: string; matched: boolean; score: number | null; clip_path: string | null; skip_reason: string | null; source_type?: 'solo_fancam' | 'group_stage' | 'solo_other' | null; crop_mode?: string | null }[]
  } | null>(null)
  // Bumps when a job transitions to done so the RecentClips panel refreshes
  // immediately instead of waiting for its next poll tick.
  const [clipsRefresh, setClipsRefresh] = useState(0)

  useEffect(() => {
    // Read the persisted job once, resolve groups list + saved-job status +
    // the global job list (so jobs started from elsewhere — another tab,
    // curl, the Admin UI — are still surfaced here), then commit state in
    // a single pass.
    let saved: SavedJob | null = null
    try {
      const raw = localStorage.getItem(SAVED_JOB_KEY)
      if (raw) saved = JSON.parse(raw) as SavedJob
    } catch {}

    const groupsP = api.listGroups()
    const savedJobP = saved
      ? api.getJob(saved.jobId).catch(() => null)  // null → vanished
      : Promise.resolve(null)
    const jobsListP = api.listJobs().catch(() => [] as JobOut[])

    Promise.all([groupsP, savedJobP, jobsListP]).then(([gs, savedJ, allJobs]) => {
      setGroups(gs)

      // Prefer the saved job when it's still alive — it has the full form
      // context. Otherwise, if there's a running oneshot out there (e.g.
      // started via curl), adopt it so the user sees live progress.
      const runningOneshot = allJobs
        .filter(j => j.kind === 'oneshot' && j.status === 'running')
        .sort((a, b) => b.started_at - a.started_at)[0]

      if (saved && savedJ) {
        restoringRef.current = saved
        setCount(saved.count)
        setAutoDelogo(saved.autoDelogo)
        setForceLandscape(!!saved.forceLandscape)
        if (typeof saved.mergeSources === 'number' && saved.mergeSources >= 2) {
          setMergeEnabled(true)
          setMergeSources(saved.mergeSources)
        }
        if (saved.mergeStyle === 'hard_cut' || saved.mergeStyle === 'xfade') {
          setMergeStyle(saved.mergeStyle)
        }
        if (typeof saved.usePose === 'boolean') {
          setUsePose(saved.usePose)
        }
        setSong(saved.song)
        setMemberLat(saved.memberLat)
        setMemberHan(saved.memberHan)
        setMemberChi(saved.memberChi)
        setGroup(saved.group)
        setJob(savedJ)
        if (savedJ.status === 'done' || savedJ.status === 'failed') {
          try { localStorage.removeItem(SAVED_JOB_KEY) } catch {}
        }
        return
      }

      // Saved job vanished — clean localStorage.
      if (saved && !savedJ) {
        try { localStorage.removeItem(SAVED_JOB_KEY) } catch {}
      }

      if (runningOneshot) {
        // Externally-started job: surface it without touching localStorage
        // (we don't know the form values, so we can't rehydrate them).
        // JobLog will stream progress from the backend.
        setJob(runningOneshot)
      }
      if (gs.length) setGroup(gs[0].name)
    }).catch(e => setErr(String(e)))
  }, [])

  useEffect(() => {
    if (!group) { setDetail(null); setMembers([]); setMemberIdx(-1); return }
    const restore = restoringRef.current
    api.getGroup(group).then(d => {
      setDetail(d)
      if (!restore) {
        const first = d.tracks.find(tr => tr.has_mp3)
        setSong(first?.title ?? '')
      }
    }).catch(e => setErr(String(e)))

    setMembersLoading(true)
    setMembers([])
    if (!restore) {
      setMemberIdx(-1)
      setMemberLat('')
      setMemberHan('')
      setMemberChi('')
    }
    api.getMembers(group).then(ms => {
      setMembers(ms)
      if (restore) {
        // Align the <select>'s index with the restored latin name.
        const idx = ms.findIndex(m => m.latin === restore.memberLat)
        setMemberIdx(idx >= 0 ? idx : -1)
      } else if (ms.length > 0) {
        setMemberIdx(0)
        setMemberLat(ms[0].latin)
        setMemberHan(ms[0].hangul)
        setMemberChi(ms[0].chinese || '')
      }
    }).catch(e => setErr(String(e)))
      .finally(() => {
        setMembersLoading(false)
        // One-shot restore; any future group change should behave normally.
        restoringRef.current = null
      })
  }, [group])

  const onPickMember = (idx: number) => {
    setMemberIdx(idx)
    if (idx >= 0 && members[idx]) {
      setMemberLat(members[idx].latin)
      setMemberHan(members[idx].hangul)
      setMemberChi(members[idx].chinese || '')
    } else {
      setMemberLat('')
      setMemberHan('')
      setMemberChi('')
    }
  }

  const sanitizeFilename = (s: string) =>
    s.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/\s+/g, ' ').trim()

  const buildDownloadName = (videoId: string): string => {
    const g = summary?.group ?? ''
    const s = summary?.song ?? ''
    const lat = memberLat.trim()
    const han = memberHan.trim()
    const chi = memberChi.trim()
    const headParts = [g, s, lat].filter(Boolean)
    const tagParts = [g, lat, han, chi].filter(Boolean).map(p => `#${p}`)
    const base = sanitizeFilename([headParts.join(' '), tagParts.join(' ')].filter(Boolean).join(' '))
    const suffix = matchedClips.length > 1 ? ` (${videoId})` : ''
    return `${base}${suffix}.mp4`
  }

  const onJobDone = async () => {
    try {
      const logs = await api.getJobLogs(job!.id, 0)
      const last = [...logs.lines].reverse().find(l => l.startsWith('ONESHOT '))
      if (last) setSummary(JSON.parse(last.slice('ONESHOT '.length)))
    } catch {}
    // Job finished — drop the saved-job breadcrumb so a later refresh
    // doesn't keep asking the backend about a done job.
    try { localStorage.removeItem(SAVED_JOB_KEY) } catch {}
    // New clips likely just landed — poke RecentClips to refresh now.
    setClipsRefresh(n => n + 1)
  }

  const go = async () => {
    setSummary(null)
    setProgress(null)
    setErr(null)
    try {
      const j = await api.oneshot({
        group, song,
        member_lat: memberLat.trim(),
        member_han: memberHan.trim() || undefined,
        count,
        delogo_corners: autoDelogo ? ['auto'] : [],
        force_landscape: forceLandscape,
        merge_sources: mergeEnabled ? mergeSources : 1,
        // Only send merge_style + use_pose when merge is actually on —
        // backend ignores them for merge_sources==1 but sending them
        // would make the job log noisier without effect.
        ...(mergeEnabled ? { merge_style: mergeStyle } : {}),
        ...(mergeEnabled && usePose ? { use_pose: true } : {}),
      })
      setJob(j)
      const saved: SavedJob = {
        jobId: j.id, group, song,
        memberLat: memberLat.trim(),
        memberHan: memberHan.trim(),
        memberChi: memberChi.trim(),
        count, autoDelogo, forceLandscape,
        mergeSources: mergeEnabled ? mergeSources : 1,
        mergeStyle: mergeEnabled ? mergeStyle : undefined,
        usePose: mergeEnabled ? usePose : undefined,
      }
      try { localStorage.setItem(SAVED_JOB_KEY, JSON.stringify(saved)) } catch {}
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const songOptions = detail?.tracks.filter(tr => tr.has_mp3) ?? []
  const canGo = !!(group && song && memberLat.trim())

  const matchedClips = summary?.clips.filter(c => c.matched) ?? []

  return (
    <div className="quick-page">
      <header className="quick-hero">
        <h1>{t('quickTitle')}</h1>
        <p>{t('quickIntro')}</p>
      </header>

      {err && <div className="quick-error">{err}</div>}

      <div className="quick-body">
      <div className="card quick-form-card">
        <div className="quick-form">
          <label>{t('quickGroup')}</label>
          <div className="select-wrap">
            <select value={group} onChange={e => setGroup(e.target.value)}>
              {groups.length === 0 && <option value="">{t('quickPickGroup')}</option>}
              {groups.map(g => <option key={g.name} value={g.name}>{g.name}</option>)}
            </select>
          </div>

          <label>{t('quickSong')}</label>
          <div className="select-wrap">
            <select value={song} onChange={e => setSong(e.target.value)}
                    disabled={songOptions.length === 0}>
              {songOptions.length === 0 && <option value="">{t('quickNoSongs')}</option>}
              {songOptions.map(tr => <option key={tr.title} value={tr.title}>{tr.title}</option>)}
            </select>
          </div>

          <label>{t('quickMember')}</label>
          {members.length > 0 ? (
            <div className="select-wrap">
              <select value={memberIdx} onChange={e => onPickMember(Number(e.target.value))}>
                {members.map((m, i) => (
                  <option key={i} value={i}>
                    {m.latin || m.hangul}{m.latin && m.hangul ? ` · ${m.hangul}` : ''}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <input value={memberLat} onChange={e => setMemberLat(e.target.value)}
                   placeholder={membersLoading ? t('quickMembersLoading') : 'Wonyoung'} />
          )}

          <label>{t('quickCount')}</label>
          <input className="count-input" type="number" min={1} max={10} value={count}
                 onChange={e => setCount(Math.max(1, Math.min(10, Number(e.target.value) || 1)))} />

          <label>{t('quickDelogo')}</label>
          <label className="check-row">
            <input type="checkbox" checked={autoDelogo}
                   onChange={e => setAutoDelogo(e.target.checked)} />
            <span>{t('quickDelogoHint')}</span>
          </label>

          <label>{t('quickForceLandscape')}</label>
          <label className="check-row">
            <input type="checkbox" checked={forceLandscape}
                   onChange={e => setForceLandscape(e.target.checked)} />
            <span>{t('quickForceLandscapeHint')}</span>
          </label>

          <label>{t('quickMerge')}</label>
          <label className="check-row">
            <input type="checkbox" checked={mergeEnabled}
                   onChange={e => setMergeEnabled(e.target.checked)} />
            <span>{t('quickMergeHint')}</span>
          </label>

          {mergeEnabled && (
            <>
              <label>{t('quickMergeN')}</label>
              <input className="count-input" type="number" min={2} max={6}
                     value={mergeSources}
                     onChange={e => setMergeSources(
                       Math.max(2, Math.min(6, Number(e.target.value) || 2)))} />

              <label>{t('quickMergeStyle')}</label>
              <div className="merge-style-row">
                <label className="check-row">
                  <input type="radio" name="mergeStyle" value="xfade"
                         checked={mergeStyle === 'xfade'}
                         onChange={() => setMergeStyle('xfade')} />
                  <span>{t('quickMergeStyleXfade')}</span>
                </label>
                <label className="check-row">
                  <input type="radio" name="mergeStyle" value="hard_cut"
                         checked={mergeStyle === 'hard_cut'}
                         onChange={() => setMergeStyle('hard_cut')} />
                  <span>{t('quickMergeStyleHard')}</span>
                </label>
                <div className="merge-style-hint">{t('quickMergeStyleHint')}</div>
              </div>

              <label>{t('quickPose')}</label>
              <label className="check-row">
                <input type="checkbox" checked={usePose}
                       onChange={e => setUsePose(e.target.checked)} />
                <span>{t('quickPoseHint')}</span>
              </label>
            </>
          )}
        </div>
        <div className="quick-cta-row">
          <button className="cta"
                  onClick={go} disabled={!canGo || (!!job && job.status === 'running')}>
            {job && job.status === 'running' ? t('quickRunning') : t('quickGo')}
          </button>
        </div>
      </div>

      <div className="quick-right">
      {summary && (
        <div className="card">
          <div className="summary-head">
            {t('quickMatched', { n: summary.matched, total: summary.considered })}
          </div>
          <ul className="summary-list">
            {summary.clips.map((c, i) => (
              <li key={i} className={c.matched ? '' : 'muted'}>
                <span className={`badge ${c.matched ? 'ok' : 'warn'}`}>
                  {c.matched ? t('quickClipLabel') : t('quickFailLabel')}
                </span>
                <span className="summary-title">{c.video_title}</span>
                {c.source_type === 'solo_fancam' && (
                  <span className="summary-score">{t('quickSourceSolo')}</span>
                )}
                {c.source_type === 'group_stage' && (
                  <span className="summary-score">{t('quickSourceStage')}</span>
                )}
                {c.score != null && <span className="summary-score">{t('quickScore')} {c.score.toFixed(3)}</span>}
                {!c.matched && c.skip_reason && <span className="summary-score">{c.skip_reason}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {job && progress && progress.phase !== 'done' && (
        <div className="card progress-card">
          <ProgressBar p={progress} t={t} />
        </div>
      )}
      {job && (
        <details className="card log-card" open={!summary}>
          <summary className="log-summary">{t('quickResultTitle')}</summary>
          <JobLog jobId={job.id} onDone={onJobDone} onProgress={setProgress} />
        </details>
      )}

      {summary && matchedClips.length > 0 && (
        <div className="card">
          <h3>{summary.group} · {summary.song}</h3>
          <div className="clip-grid">
            {matchedClips.map(c => {
              const url = api.clipUrl(summary.group, summary.song, c.video_id)
              const dlUrl = api.clipDownloadUrl(summary.group, summary.song, c.video_id)
              const srcUrl = api.videoUrl(c.video_stem)
              const fname = buildDownloadName(c.video_id)
              const onDownload = () => {
                // Best-effort: also POST the keep marker so the sidecar
                // is created even if the server ignored the ?keep=1 query.
                api.markClipKept(summary.group, summary.song, c.video_id).catch(() => {})
              }
              return (
                <div key={c.video_id} className="clip-item">
                  <div className="clip-compare">
                    <figure>
                      <figcaption>{t('quickClipProduced')}</figcaption>
                      <video controls preload="metadata" src={url} />
                    </figure>
                    <figure>
                      <figcaption>{t('quickClipSource')}</figcaption>
                      <video controls preload="metadata" src={srcUrl} />
                    </figure>
                  </div>
                  <div className="clip-actions">
                    <a className="btn-dl" href={dlUrl} download={fname} onClick={onDownload}>{t('quickDownload')}</a>
                    {c.score != null && <span className="clip-meta">{t('quickScore')} {c.score.toFixed(3)}</span>}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
      </div>

      <div style={{ gridColumn: '1 / -1' }}>
        <JobsHistory />
      </div>
      <div style={{ gridColumn: '1 / -1' }}>
        <RecentClips refreshToken={clipsRefresh} />
      </div>
      </div>
    </div>
  )
}
