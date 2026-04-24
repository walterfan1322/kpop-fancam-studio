import { useEffect, useState } from 'react'
import { api, type JobOut } from './api'
import { JobLog } from './JobLog'
import { useT } from './i18n'

/** How often we poll /api/jobs while the panel is visible. */
const POLL_MS = 3000
/** Soft cap on rendered rows — the backend already returns only the 50 most
 *  recent. Showing more than this makes the panel noisy in simple mode. */
const MAX_ROWS = 12

/** Parse a `--flag value` style argv-list into a {flag: value} dict. We
 *  only care about oneshot_fancam.py's flags here. */
function parseArgs(args: string[] | undefined): Record<string, string> {
  if (!args) return {}
  const out: Record<string, string> = {}
  for (let i = 0; i < args.length; i++) {
    const a = args[i]
    if (a.startsWith('--') && i + 1 < args.length && !args[i + 1].startsWith('--')) {
      out[a.slice(2)] = args[i + 1]
      i++
    } else if (a.startsWith('--')) {
      out[a.slice(2)] = 'true'
    }
  }
  return out
}

/** Produce a one-line human title for a job, based on script + args.
 *  For oneshot: "IVE · After LIKE · Wonyoung". For other scripts: fall back
 *  to "<kind> <first arg>". */
function jobTitle(j: JobOut): string {
  const a = parseArgs(j.args)
  if (j.kind === 'oneshot') {
    const parts = [a.group, a.song, a['member-lat']].filter(Boolean)
    return parts.length ? parts.join(' · ') : 'oneshot'
  }
  if (j.kind === 'fetch') return `fetch · ${a.group ?? (j.args?.[0] ?? '')}`
  if (j.kind === 'download') return `download${a.group ? ' · ' + a.group : ''}`
  if (j.kind === 'resolve') return `resolve · ${a.group ?? ''}`
  if (j.kind === 'match') return `match · ${a.group ?? ''}`
  if (j.kind === 'probe') return 'probe'
  return j.kind
}

/** Human-relative "5s ago" / "running 12s". */
function relTime(ts: number, now: number): string {
  const s = Math.max(0, Math.round(now - ts))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  return `${h}h`
}

function statusIcon(status: JobOut['status'], rc: number | null): string {
  if (status === 'running' || status === 'pending') return '⏳'
  if (status === 'done' && rc === 0) return '✅'
  return '❌'
}

export function JobsHistory() {
  const t = useT()
  const [jobs, setJobs] = useState<JobOut[] | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [now, setNow] = useState(() => Date.now() / 1000)
  const [err, setErr] = useState<string | null>(null)
  const [open, setOpen] = useState(true)

  useEffect(() => {
    let alive = true
    let timer: number | undefined

    const tick = async () => {
      try {
        const list = await api.listJobs()
        if (!alive) return
        setJobs(list)
        setNow(Date.now() / 1000)
        setErr(null)
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e))
      } finally {
        if (alive) timer = window.setTimeout(tick, POLL_MS)
      }
    }
    tick()

    // Also tick a local "now" every second so the relative-time labels on
    // running jobs don't look frozen between polls.
    const nowTimer = window.setInterval(() => setNow(Date.now() / 1000), 1000)

    return () => {
      alive = false
      if (timer) window.clearTimeout(timer)
      window.clearInterval(nowTimer)
    }
  }, [])

  const rows = (jobs ?? []).slice(0, MAX_ROWS)
  const running = rows.filter(j => j.status === 'running' || j.status === 'pending').length

  return (
    <div className="card jobs-panel">
      <div className="jobs-header" onClick={() => setOpen(o => !o)}>
        <span className="jobs-title">
          📋 {t('jobsTitle')}
          {running > 0 && <span className="jobs-running-pill">{running} {t('jobsRunning')}</span>}
        </span>
        <span className="jobs-chevron">{open ? '▾' : '▸'}</span>
      </div>
      {open && (
        <>
          {err && <div className="jobs-err">{err}</div>}
          {jobs === null ? (
            <div className="jobs-empty">{t('loading')}</div>
          ) : rows.length === 0 ? (
            <div className="jobs-empty">{t('jobsEmpty')}</div>
          ) : (
            <ul className="jobs-list">
              {rows.map(j => {
                const isExpanded = expandedId === j.id
                const isActive = j.status === 'running' || j.status === 'pending'
                const endedAt = j.finished_at > 0 ? j.finished_at : now
                const label = isActive
                  ? `${t('jobsRunningFor')} ${relTime(j.started_at, now)}`
                  : `${relTime(endedAt, now)} ${t('jobsAgo')}`
                return (
                  <li key={j.id}
                      className={`jobs-row ${isActive ? 'active' : ''} ${isExpanded ? 'expanded' : ''}`}>
                    <button
                      className="jobs-row-head"
                      onClick={() => setExpandedId(isExpanded ? null : j.id)}
                    >
                      <span className="jobs-icon">
                        {statusIcon(j.status, j.return_code)}
                      </span>
                      <span className="jobs-kind">{j.kind}</span>
                      <span className="jobs-row-title">{jobTitle(j)}</span>
                      <span className="jobs-time">{label}</span>
                      <span className="jobs-chevron">{isExpanded ? '▾' : '▸'}</span>
                    </button>
                    {isExpanded && (
                      <div className="jobs-row-body">
                        <JobLog jobId={j.id} />
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </>
      )}
    </div>
  )
}
