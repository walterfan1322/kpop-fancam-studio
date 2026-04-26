import { useEffect, useState, type MouseEvent } from 'react'
import { api, type ClipOut } from './api'
import { useT } from './i18n'

/** Poll interval for the clips list — pick up clips produced by the
 *  currently-running job, or by another tab / curl, without requiring a
 *  manual refresh. Kept slow enough not to hammer the filesystem. */
const POLL_MS = 15000
/** How many clips to surface. More than this and the panel starts scrolling
 *  on a typical laptop — if the user wants the full archive they can flip
 *  to Admin → Videos & Matching for the complete list. */
const MAX_ITEMS = 6

function relTime(ts: number, now: number): string {
  if (!ts) return ''
  const s = Math.max(0, Math.round(now - ts))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 48) return `${h}h`
  const d = Math.floor(h / 24)
  return `${d}d`
}

export function RecentClips({ refreshToken }: { refreshToken: number }) {
  const t = useT()
  const [clips, setClips] = useState<ClipOut[] | null>(null)
  const [now, setNow] = useState(() => Date.now() / 1000)
  const [open, setOpen] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  // Optimistic-hide set: keys we've told the server to hide, used so
  // the card disappears immediately rather than waiting for the next
  // poll. The persistent source of truth is the `.mp4.hidden` sidecar
  // on the server — list_clips skips hidden entries, so after the
  // backend acknowledges and the next poll runs, the clip will already
  // be gone from `clips`, and this set is a no-op. The underlying mp4
  // is preserved on disk (not deleted).
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  useEffect(() => {
    let alive = true
    let timer: number | undefined

    const tick = async () => {
      try {
        const list = await api.listClips()
        if (!alive) return
        // Most-recent-first by mtime. Older backends may return mtime=0 /
        // undefined — fall back to stable ordering by name so the list
        // doesn't flicker.
        const sorted = [...list].sort((a, b) => (b.mtime ?? 0) - (a.mtime ?? 0))
        setClips(sorted)
        setNow(Date.now() / 1000)
        setErr(null)
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e))
      } finally {
        if (alive) timer = window.setTimeout(tick, POLL_MS)
      }
    }
    tick()

    return () => {
      alive = false
      if (timer) window.clearTimeout(timer)
    }
    // refreshToken bumps when a job completes — re-run the fetch immediately
    // rather than waiting for the next poll.
  }, [refreshToken])

  // Drop dismissed BEFORE slicing so a user-hidden clip lets the next
  // one slide into view (the grid still shows up to MAX_ITEMS).
  const rows = (clips ?? [])
    .filter(c => !dismissed.has(`${c.group}/${c.song}/${c.title}`))
    .slice(0, MAX_ITEMS)

  return (
    <div className="card recent-clips-panel">
      <div className="recent-clips-header" onClick={() => setOpen(o => !o)}>
        <span className="recent-clips-title">
          🎬 {t('recentClipsTitle')}
          {clips && clips.length > 0 && (
            <span className="recent-clips-count">{clips.length}</span>
          )}
        </span>
        <span className="jobs-chevron">{open ? '▾' : '▸'}</span>
      </div>
      {open && (
        <>
          {err && <div className="jobs-err">{err}</div>}
          {clips === null ? (
            <div className="jobs-empty">{t('loading')}</div>
          ) : rows.length === 0 ? (
            <div className="jobs-empty">{t('recentClipsEmpty')}</div>
          ) : (
            <div className="recent-clips-grid">
              {rows.map(c => {
                const url = api.clipUrl(c.group, c.song, c.title)
                // Earlier versions used `${url}#t=0.1` + preload="auto" to
                // show a non-black thumbnail without a click. That combo
                // raced badly: Chrome would fetch the moov, then queue the
                // seek-to-0.1 byte-range fetch behind the other cards'
                // preloads. The element sat on a black canvas and the
                // first ▶ click was no-op because the pending seek blocked
                // play(). preload="metadata" is strictly more reliable —
                // browser fetches just the moov, surfaces real controls,
                // and starts streaming the moment the user clicks ▶.
                const dlUrl = api.clipDownloadUrl(c.group, c.song, c.title)
                const rel = relTime(c.mtime ?? 0, now)
                const fname = `${c.group} ${c.song} (${c.title}).mp4`
                const onDownload = () => {
                  api.markClipKept(c.group, c.song, c.title).catch(() => {})
                }
                const key = `${c.group}/${c.song}/${c.title}`
                const onDismiss = (ev: MouseEvent) => {
                  // Prevent the click from reaching the <video> (which
                  // would toggle play) when users aim at the X.
                  ev.stopPropagation()
                  // Optimistic UI first, server call second. If the call
                  // fails we revert so the card reappears — don't want
                  // the user to think they hid something that's actually
                  // still listed server-side.
                  setDismissed(prev => {
                    const next = new Set(prev)
                    next.add(key)
                    return next
                  })
                  api.markClipHidden(c.group, c.song, c.title).catch(err => {
                    console.warn('[RecentClips] hide failed, restoring', err)
                    setDismissed(prev => {
                      const next = new Set(prev)
                      next.delete(key)
                      return next
                    })
                  })
                }
                return (
                  <div key={key} className="recent-clip-item">
                    <button type="button" className="recent-clip-dismiss"
                            onClick={onDismiss}
                            title={t('recentClipsDismiss')}
                            aria-label={t('recentClipsDismiss')}>
                      ×
                    </button>
                    <video controls preload="metadata" src={url} />
                    <div className="recent-clip-meta">
                      <div className="recent-clip-top">
                        <span className="recent-clip-group">{c.group}</span>
                        <span className="recent-clip-song">{c.song}</span>
                      </div>
                      <div className="recent-clip-bot">
                        {c.size_mb != null && (
                          <span className="recent-clip-size">{c.size_mb.toFixed(1)} MB</span>
                        )}
                        {rel && <span className="recent-clip-time">{rel}</span>}
                        <a className="btn-dl btn-dl-sm"
                           href={dlUrl} download={fname} onClick={onDownload}>
                          {t('quickDownload')}
                        </a>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </>
      )}
    </div>
  )
}
