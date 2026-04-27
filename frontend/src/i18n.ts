import { createContext, useContext, useEffect, useState } from 'react'

export type Lang = 'en' | 'zh'

const STORAGE_KEY = 'lang'

const dict = {
  // App shell
  appTitle: { en: '🎵 K-pop TikTok', zh: '🎵 K-pop TikTok' },
  groups: { en: 'Groups', zh: '團體' },
  loading: { en: 'loading…', zh: '載入中…' },
  noGroupsYet: { en: 'No groups yet. Add one below.', zh: '尚無團體，請於下方新增。' },
  mp3sCount: { en: 'mp3s', zh: '首' },
  addGroup: { en: 'Add group', zh: '新增團體' },
  groupPlaceholder: { en: 'IVE / aespa / NewJeans', zh: 'IVE / aespa / NewJeans' },
  limit: { en: 'Limit', zh: '筆數' },
  fetchMB: { en: '🔍 Fetch from MusicBrainz', zh: '🔍 從 MusicBrainz 擷取' },
  selectGroupPrompt: { en: 'Select a group, or add one on the left.', zh: '請從左側選擇或新增團體。' },

  // Group detail
  artist: { en: 'Artist', zh: '藝人' },
  tracks: { en: 'Tracks', zh: '曲目' },
  downloaded: { en: 'Downloaded', zh: '已下載' },
  coverage: { en: 'Coverage', zh: '涵蓋率' },
  tabTracks: { en: 'Tracks', zh: '曲目' },
  tabResolve: { en: 'Resolve TikTok URLs', zh: '解析 TikTok 連結' },
  tabVideos: { en: 'Videos & Matching', zh: '影片與比對' },
  saveChanges: { en: '💾 Save changes', zh: '💾 儲存變更' },
  addRow: { en: '+ Add row', zh: '+ 新增一列' },
  downloadThisGroup: { en: '⬇️ Download this group', zh: '⬇️ 下載此團體' },
  deleteGroup: { en: 'Delete group', zh: '刪除團體' },
  title: { en: 'Title', zh: '曲名' },
  status: { en: 'Status', zh: '狀態' },
  musicUrl: { en: 'Music URL', zh: '音樂 URL' },
  preview: { en: 'Preview', zh: '預覽' },
  badgeOk: { en: 'OK', zh: 'OK' },
  badgeNoAudio: { en: 'no audio', zh: '無音訊' },
  badgeNotFound: { en: 'not found', zh: '找不到' },
  badgePending: { en: 'pending', zh: '待處理' },
  confirmDeleteGroup: {
    en: 'Delete group {name}? Only the YAML entry will be removed; mp3 files are kept.',
    zh: '刪除團體 {name}？只會刪 yaml 條目，不會刪 mp3 檔。',
  },

  // Resolve
  pasteTiktokUrls: { en: 'Paste TikTok share URLs', zh: '貼上 TikTok 分享連結' },
  resolveHelp: {
    en: "Each URL (vt.tiktok.com/... or tiktok.com/music/...) is resolved to a music page, then fuzzy-matched against this group's tracks. New titles are added automatically.",
    zh: '每一條 URL (vt.tiktok.com/... 或 tiktok.com/music/...) 會被解析到音樂頁，再與此團體的曲目做模糊比對；找不到的曲名會自動新增。',
  },
  resolveMerge: { en: 'Resolve & merge', zh: '解析並合併' },

  // Videos tab
  searchFancams: { en: 'Search fancams (auto)', zh: '自動搜尋直拍' },
  memberLatinPh: { en: 'Member name (Latin, e.g. Wonyoung)', zh: '成員名 (拉丁, 例: Wonyoung)' },
  memberHangulPh: { en: 'Member name (Hangul, e.g. 장원영) — optional', zh: '成員名 (韓文, 例: 장원영) — 可選' },
  resultsPerQuery: { en: 'results/query', zh: '每筆查詢結果' },
  minHeight: { en: 'min height', zh: '最低解析度' },
  minViews: { en: 'min views', zh: '最低觀看數' },
  searching: { en: 'Searching…', zh: '搜尋中…' },
  search: { en: '🔎 Search', zh: '🔎 搜尋' },
  passedTotal: { en: '{p} passed / {t} total', zh: '{p} 通過 / 共 {t}' },
  downloadSelected: { en: '⬇️ Download {n} selected', zh: '⬇️ 下載已選 {n} 支' },
  colTitle: { en: 'Title', zh: '標題' },
  colChannel: { en: 'Channel', zh: '頻道' },
  colDur: { en: 'Dur', zh: '秒數' },
  colRes: { en: 'Res', zh: '解析度' },
  colViews: { en: 'Views', zh: '觀看' },
  colStatus: { en: 'Status', zh: '狀態' },
  badgeHave: { en: 'have', zh: '已有' },
  badgePass: { en: 'pass', zh: '通過' },

  downloadByUrl: { en: 'Download by URL', zh: '以 URL 下載' },
  download: { en: '⬇️ Download', zh: '⬇️ 下載' },

  videosHeader: { en: 'Videos ({n}) — sorted by quality', zh: '影片 ({n}) — 依品質排序' },
  threshold: { en: 'Threshold', zh: '門檻' },
  marginLbl: { en: 'Margin', zh: 'Margin' },
  marginHint: { en: 'Top1 - Top2 score must exceed this', zh: 'Top1 與 Top2 的分差需大於此值' },
  matchesAgainst: { en: 'Matches against group', zh: '比對團體' },
  noVideos: { en: 'No videos yet.', zh: '尚無影片。' },
  colSize: { en: 'Size', zh: '大小' },
  colQuality: { en: 'Quality', zh: '品質' },
  colActions: { en: 'Actions', zh: '操作' },
  source: { en: 'source', zh: '來源' },
  cutsPerSec: { en: 'cuts', zh: '切換' },
  persons: { en: 'persons', zh: '人數' },
  heightShort: { en: 'h', zh: '高' },
  lateralShort: { en: 'Δx', zh: 'Δx' },
  notProbed: { en: 'not probed', zh: '尚未分析' },
  probe: { en: '📊 Probe', zh: '📊 分析' },
  probeTitle: { en: 'Quality probe (cuts + YOLO people)', zh: '品質分析 (切換頻率 + YOLO 人物)' },
  match: { en: '🎯 Match', zh: '🎯 比對' },
  confirmDeleteVideo: { en: 'Delete video {name}?', zh: '刪除影片 {name}？' },

  clipsHeader: { en: 'Clips for {group} ({n})', zh: '{group} 的片段 ({n})' },
  noClipsYet: { en: 'No clips yet — run Match on a video.', zh: '尚無片段 — 對影片跑 Match 即可產生。' },
  colSong: { en: 'Song', zh: '歌曲' },
  confirmDeleteClip: { en: 'Delete clip {name}?', zh: '刪除片段 {name}？' },

  job: { en: 'Job', zh: '任務' },

  // Language toggle
  langEn: { en: 'EN', zh: 'EN' },
  langZh: { en: '中', zh: '中' },

  // Mode toggle
  modeSimple: { en: 'Simple', zh: '簡易' },
  modeAdmin: { en: 'Admin', zh: '進階' },

  // Quick mode
  quickTitle: { en: 'Generate a fancam', zh: '產生 Fancam' },
  quickIntro: {
    en: 'Pick a song + a member + how many videos. We search, download, match, and extract clips aligned to that song.',
    zh: '選一首歌、一位成員、幾支影片。系統會搜尋、下載、比對音訊並切出對應片段。',
  },
  quickGroup: { en: 'Group', zh: '團體' },
  quickSong: { en: 'Song', zh: '歌曲' },
  quickMemberLat: { en: 'Member (Latin)', zh: '成員 (拉丁)' },
  quickMemberHan: { en: 'Member (Hangul, optional)', zh: '成員 (韓文, 可選)' },
  quickMember: { en: 'Member', zh: '成員' },
  quickMembersLoading: { en: 'Loading members…', zh: '載入成員中…' },
  quickCount: { en: 'How many videos', zh: '影片數量' },
  quickDelogo: { en: 'Watermark', zh: '浮水印' },
  quickDelogoHint: { en: 'Auto-detect and remove corner logos (M2, ZOOM, etc.)', zh: '自動偵測並去除角落 logo (M2、ZOOM 等)' },
  quickForceLandscape: { en: 'Orientation', zh: '影片方向' },
  quickForceLandscapeHint: { en: 'Force landscape (16:9 stage / music-show videos only)', zh: '強制橫向影片 (只選 16:9 團體舞台 / 音樂節目)' },
  quickMerge: { en: 'Multi-angle merge', zh: '多機合併' },
  quickMergeHint: { en: 'Fuse multiple sources into one clip, swapping to whichever angle is showing the member', zh: '將多部同首歌的影片合併成一支，缺人時自動切到別機有人的那段' },
  quickMergeN: { en: 'Max sources', zh: '最多機位數' },
  quickMergeStyle: { en: 'Cut style', zh: '切換方式' },
  quickMergeStyleXfade: { en: 'Cross-fade (0.5s, safer)', zh: '淡入淡出 (0.5 秒，較安全)' },
  quickMergeStyleHard: { en: 'Hard cut (outfit-swap look)', zh: '直接硬切 (瞬間換裝風格)' },
  quickMergeStyleHint: { en: 'Hard cut only works well when sources share the same angle/framing', zh: '硬切只在各來源角度框取一致時才好看' },
  quickPose: { en: 'Pose-guided framing', zh: '姿態對齊取景' },
  quickPoseHint: { en: 'Use pose keypoints to lock head position and prefer same-angle cuts (adds ~30s/source of CPU inference)', zh: '用姿態關鍵點鎖定頭部位置並偏好同角度切換 (每部來源多耗 ~30 秒 CPU)' },
  quickRotation: { en: 'Outfit-swap cadence (s, min–max)', zh: '換裝節奏 (秒，最少–最多)' },
  quickRotationMin: { en: 'Minimum slot duration (seconds)', zh: '單段最短秒數' },
  quickRotationMax: { en: 'Maximum slot duration (seconds, set above min for variable cadence)', zh: '單段最長秒數 (大於最小值才會啟用變動節奏)' },
  quickRotationHint: { en: '0 = greedy (best source dominates). Set both fields >0 (e.g. 4 and 8) to force source-swap every 4–8s — visible outfit/stage changes. Equal values give a fixed cadence; unequal values give variable cadence drawn from [min, max]. Pair with hard cut.', zh: '0 = 貪婪模式 (最佳來源主導)。兩格都填 > 0 (例如 4 和 8) 會強制每 4–8 秒換來源——明顯換裝/換舞台。相等 = 固定節奏；不等 = 在 [最少, 最多] 區間隨機。建議搭配硬切。' },
  quickGo: { en: '✨ Generate', zh: '✨ 產生' },
  quickRunning: { en: 'Running…', zh: '執行中…' },
  quickNoSongs: { en: 'This group has no downloaded mp3s yet.', zh: '此團體尚未下載任何 mp3。' },
  quickPickGroup: { en: 'Pick a group', zh: '選擇團體' },
  quickResultTitle: { en: 'Result', zh: '結果' },
  quickMatched: { en: '{n} of {total} matched', zh: '{total} 支中有 {n} 支配對成功' },
  quickClipLabel: { en: 'Clip', zh: '片段' },
  quickFailLabel: { en: 'Skipped', zh: '未採用' },
  quickScore: { en: 'score', zh: '分數' },
  quickDownload: { en: '⬇ Download', zh: '⬇ 下載' },
  quickClipProduced: { en: 'Produced', zh: '產出片段' },
  quickClipSource: { en: 'Original', zh: '原片' },
  quickSourceSolo: { en: 'solo fancam', zh: '個人直拍' },
  quickSourceStage: { en: 'group stage', zh: '團體舞台' },
  quickSourceOther: { en: 'other', zh: '其他' },
  progSearching: { en: 'Searching YouTube…', zh: '搜尋中…' },
  progPicking:   { en: 'Preparing candidates…', zh: '準備候選影片…' },
  progWorking:   { en: 'Working…', zh: '處理中…' },
  progDone:      { en: 'Done', zh: '完成' },
  progMatched:   { en: 'matched', zh: '已產出' },

  // Jobs panel
  jobsTitle:      { en: 'Jobs',         zh: '任務紀錄' },
  jobsRunning:    { en: 'running',      zh: '執行中' },
  jobsEmpty:      { en: 'No jobs yet.', zh: '尚無任務。' },
  jobsRunningFor: { en: 'running',      zh: '已執行' },
  jobsAgo:        { en: 'ago',          zh: '前' },

  // Recent clips panel (Quick Mode)
  recentClipsTitle: { en: 'Recent clips',          zh: '最近產出的片段' },
  recentClipsEmpty: { en: 'No clips produced yet.', zh: '尚未產出任何片段。' },
  recentClipsDismiss: { en: 'Hide from this list',  zh: '從列表中隱藏' },
} as const

export type StringKey = keyof typeof dict

export function loadLang(): Lang {
  const stored = typeof localStorage !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null
  return stored === 'zh' ? 'zh' : 'en'
}

export const LangContext = createContext<{ lang: Lang; setLang: (l: Lang) => void }>({
  lang: 'en',
  setLang: () => {},
})

export function useLang() {
  return useContext(LangContext)
}

export function useLangState() {
  const [lang, setLang] = useState<Lang>(loadLang)
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, lang)
  }, [lang])
  return { lang, setLang }
}

export function t(lang: Lang, key: StringKey, vars?: Record<string, string | number>): string {
  let s: string = dict[key][lang]
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      s = s.replace(`{${k}}`, String(v))
    }
  }
  return s
}

export function useT() {
  const { lang } = useLang()
  return (key: StringKey, vars?: Record<string, string | number>) => t(lang, key, vars)
}
