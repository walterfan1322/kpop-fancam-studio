import { useEffect, useRef, useState } from 'react'
import { api, type JobOut } from './api'

export type OneshotProgress = {
  phase: 'searching' | 'picking' | 'working' | 'done'
  pool: number       // total attempts the run will make
  done: number       // attempts that finished (matched or skipped)
  matched: number    // attempts that produced a clip
  target: number     // user-requested clip count
  current?: string   // short description of what's happening right now
}

type Props = {
  jobId: string
  onDone?: (rc: number | null) => void
  onProgress?: (p: OneshotProgress) => void
}

// Polls /api/jobs/{id}/logs since=offset every 400ms. Simpler than SSE and
// avoids EventSource quirks; fine for a single-user tool.
// Pure function: fold a chunk of new log lines into an OneshotProgress state.
// Exported so QuickMode can re-derive without re-parsing everything.
export function reduceProgress(prev: OneshotProgress, newLines: string[]): OneshotProgress {
  const next = { ...prev }
  for (const raw of newLines) {
    const l = raw.trimEnd()
    if (l.startsWith('[search]')) {
      next.phase = 'searching'
      next.current = l.slice(9).trim() || next.current
      continue
    }
    const pickM = l.match(/^\[pick\] pool=(\d+).*target=(\d+)/)
    if (pickM) {
      next.pool = Number(pickM[1])
      next.target = Number(pickM[2])
      next.phase = 'picking'
      next.current = undefined
      continue
    }
    const dlM = l.match(/^\[download\] \((\d+)\/(\d+)\)\s+(\S+)/)
    if (dlM) {
      next.phase = 'working'
      next.current = `下載 ${dlM[1]}/${dlM[2]}`
      continue
    }
    const mtM = l.match(/^\[match\] \((\d+)\/(\d+)\)\s+(\S+)/)
    if (mtM) {
      next.phase = 'working'
      next.current = `音訊配對 ${mtM[1]}/${mtM[2]}`
      continue
    }
    if (l.startsWith('[extract]')) {
      next.matched += 1
      next.done += 1
      next.current = '切片輸出中'
      continue
    }
    if (l.startsWith('[skip:') || l.startsWith('[match:error]')) {
      next.done += 1
      continue
    }
    if (l.startsWith('[stop]') || l.startsWith('ONESHOT ')) {
      next.phase = 'done'
      next.current = undefined
      continue
    }
  }
  return next
}

const EMPTY_PROGRESS: OneshotProgress = {
  phase: 'searching', pool: 0, done: 0, matched: 0, target: 0,
}

export function JobLog({ jobId, onDone, onProgress }: Props) {
  const [lines, setLines] = useState<string[]>([])
  const [status, setStatus] = useState<JobOut['status']>('pending')
  const [rc, setRc] = useState<number | null>(null)
  const offsetRef = useRef(0)
  const doneRef = useRef(false)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const onDoneRef = useRef(onDone)
  const onProgressRef = useRef(onProgress)
  const progressRef = useRef<OneshotProgress>(EMPTY_PROGRESS)
  useEffect(() => { onDoneRef.current = onDone }, [onDone])
  useEffect(() => { onProgressRef.current = onProgress }, [onProgress])

  useEffect(() => {
    let cancelled = false
    offsetRef.current = 0
    doneRef.current = false
    progressRef.current = EMPTY_PROGRESS
    onProgressRef.current?.(EMPTY_PROGRESS)
    setLines([])
    setStatus('pending')
    setRc(null)

    let notFoundStreak = 0
    const tick = async () => {
      if (cancelled || doneRef.current) return
      try {
        const r = await api.getJobLogs(jobId, offsetRef.current)
        if (cancelled) return
        notFoundStreak = 0
        if (r.lines.length) {
          setLines(prev => [...prev, ...r.lines])
          offsetRef.current = r.next_offset
          progressRef.current = reduceProgress(progressRef.current, r.lines)
          onProgressRef.current?.(progressRef.current)
        }
        setStatus(r.status as JobOut['status'])
        setRc(r.return_code)
        if (r.status === 'done' || r.status === 'failed') {
          doneRef.current = true
          progressRef.current = { ...progressRef.current, phase: 'done', current: undefined }
          onProgressRef.current?.(progressRef.current)
          onDoneRef.current?.(r.return_code)
          return
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        // Job vanished (usually because the backend restarted mid-run).
        // Stop polling after a few 404s instead of spamming the log.
        if (msg.startsWith('404')) {
          notFoundStreak += 1
          if (notFoundStreak >= 3) {
            doneRef.current = true
            setStatus('failed')
            setLines(prev => [...prev, '[job no longer on server — backend restarted?]'])
            onDoneRef.current?.(null)
            return
          }
        } else {
          setLines(prev => [...prev, `[poll error] ${msg}`])
        }
      }
      setTimeout(tick, 400)
    }
    tick()
    return () => { cancelled = true }
  }, [jobId])

  useEffect(() => {
    if (boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight
    }
  }, [lines])

  const badgeClass =
    status === 'done' ? 'badge ok' :
    status === 'failed' ? 'badge err' :
    status === 'running' ? 'badge warn' : 'badge muted'

  return (
    <div>
      <div style={{ marginBottom: 6, display: 'flex', alignItems: 'center', gap: 10 }}>
        <span className={badgeClass}>{status}{rc != null ? ` (rc=${rc})` : ''}</span>
        <span style={{ color: 'var(--muted)', fontSize: 12 }}>job {jobId}</span>
      </div>
      <div ref={boxRef} className="log-panel">
        {lines.length === 0 ? <span style={{ color: 'var(--muted)' }}>waiting for output...</span>
          : lines.join('\n')}
      </div>
    </div>
  )
}
