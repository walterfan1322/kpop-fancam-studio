import { useCallback, useEffect, useState } from 'react'
import { api, type Candidate, type ClipOut, type JobOut, type VideoMeta } from './api'
import { JobLog } from './JobLog'
import { useT } from './i18n'

type Props = {
  groupName: string
  artistName: string
}

export function VideosTab({ groupName, artistName }: Props) {
  const t = useT()
  const [videos, setVideos] = useState<VideoMeta[]>([])
  const [clips, setClips] = useState<ClipOut[]>([])
  const [url, setUrl] = useState('')
  const [threshold, setThreshold] = useState(0.6)
  const [margin, setMargin] = useState(0.03)
  const [activeJob, setActiveJob] = useState<JobOut | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const defaultMember = ''
  const [memberLat, setMemberLat] = useState(defaultMember)
  const [memberHan, setMemberHan] = useState('')
  const [searchLimit, setSearchLimit] = useState(20)
  const [searchMinH, setSearchMinH] = useState(1080)
  const [searchMinViews, setSearchMinViews] = useState(5000)
  const [candidates, setCandidates] = useState<Candidate[] | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [searching, setSearching] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [v, c] = await Promise.all([api.listVideos(), api.listClips()])
      setVideos(v)
      setClips(c)
      setErr(null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const startDownload = async () => {
    const u = url.trim()
    if (!u) return
    const job = await api.downloadVideo(u)
    setActiveJob(job)
    setUrl('')
  }

  const startMatch = async (stem: string) => {
    const job = await api.startMatch(groupName, stem, threshold, margin)
    setActiveJob(job)
  }

  const startProbe = async (stem: string) => {
    const job = await api.probeVideo(stem)
    setActiveJob(job)
  }

  const runSearch = async () => {
    const queries: string[] = []
    if (memberLat.trim()) queries.push(`${artistName} ${memberLat.trim()} fancam`)
    if (memberLat.trim()) queries.push(`${artistName} ${memberLat.trim()} focus cam`)
    if (memberHan.trim()) queries.push(`${artistName} ${memberHan.trim()} 직캠`)
    if (queries.length === 0) return
    setSearching(true)
    setCandidates(null)
    try {
      const res = await api.searchVideos({
        queries,
        limit: searchLimit,
        min_height: searchMinH,
        min_views: searchMinViews,
        title_any: ['fancam', '직캠', 'focus', memberLat.trim(), memberHan.trim()].filter(Boolean),
      })
      setCandidates(res)
      setSelected(new Set(res.filter(r => r.passed && !r.already_downloaded).map(r => r.id)))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setSearching(false)
    }
  }

  const startBatchDownload = async () => {
    if (!candidates) return
    const urls = candidates.filter(c => selected.has(c.id)).map(c => c.url)
    if (urls.length === 0) return
    const job = await api.downloadBatch(urls)
    setActiveJob(job)
  }

  const toggle = (id: string) => {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setSelected(next)
  }

  const groupClips = clips.filter(c => c.group === groupName)

  const videosSorted = [...videos].sort((a, b) => {
    const sa = a.quality?.scores.composite ?? -1
    const sb = b.quality?.scores.composite ?? -1
    return sb - sa
  })

  return (
    <div>
      {err && <div style={{ color: 'var(--err)', fontSize: 12, marginBottom: 10 }}>{err}</div>}

      <div className="card">
        <h3>{t('searchFancams')}</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 10 }}>
          <input
            placeholder={t('memberLatinPh')}
            value={memberLat}
            onChange={e => setMemberLat(e.target.value)}
          />
          <input
            placeholder={t('memberHangulPh')}
            value={memberHan}
            onChange={e => setMemberHan(e.target.value)}
          />
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
          <span style={{ color: 'var(--muted)', fontSize: 12 }}>{t('resultsPerQuery')}</span>
          <input type="number" style={{ width: 70 }} min={1} max={50}
                 value={searchLimit} onChange={e => setSearchLimit(Number(e.target.value))} />
          <span style={{ color: 'var(--muted)', fontSize: 12 }}>{t('minHeight')}</span>
          <input type="number" style={{ width: 80 }} step={180}
                 value={searchMinH} onChange={e => setSearchMinH(Number(e.target.value))} />
          <span style={{ color: 'var(--muted)', fontSize: 12 }}>{t('minViews')}</span>
          <input type="number" style={{ width: 90 }} step={1000}
                 value={searchMinViews} onChange={e => setSearchMinViews(Number(e.target.value))} />
          <button onClick={runSearch} disabled={searching || !memberLat.trim()}>
            {searching ? t('searching') : t('search')}
          </button>
        </div>

        {candidates && (
          <>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>
              {t('passedTotal', { p: candidates.filter(c => c.passed).length, t: candidates.length })}
              {' · '}
              <button onClick={startBatchDownload} disabled={selected.size === 0}>
                {t('downloadSelected', { n: selected.size })}
              </button>
            </div>
            <table className="tracks-table">
              <thead>
                <tr>
                  <th style={{ width: 30 }}></th>
                  <th>{t('colTitle')}</th>
                  <th style={{ width: 120 }}>{t('colChannel')}</th>
                  <th style={{ width: 60 }}>{t('colDur')}</th>
                  <th style={{ width: 60 }}>{t('colRes')}</th>
                  <th style={{ width: 80 }}>{t('colViews')}</th>
                  <th style={{ width: 140 }}>{t('colStatus')}</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map(c => (
                  <tr key={c.id} style={{ opacity: c.passed ? 1 : 0.55 }}>
                    <td>
                      <input type="checkbox"
                             disabled={c.already_downloaded}
                             checked={selected.has(c.id)}
                             onChange={() => toggle(c.id)} />
                    </td>
                    <td>
                      <a href={c.url} target="_blank" rel="noreferrer">{c.title}</a>
                    </td>
                    <td style={{ fontSize: 11, color: 'var(--muted)' }}>{c.uploader}</td>
                    <td>{c.duration ? `${Math.round(c.duration)}s` : '—'}</td>
                    <td>{c.height ? `${c.height}p` : '—'}</td>
                    <td>{c.view_count?.toLocaleString() ?? '—'}</td>
                    <td style={{ fontSize: 11 }}>
                      {c.already_downloaded ? <span className="badge ok">{t('badgeHave')}</span>
                        : c.passed ? <span className="badge ok">{t('badgePass')}</span>
                        : <span className="badge warn">{c.reject_reasons.join(', ')}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      <div className="card">
        <h3>{t('downloadByUrl')}</h3>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            style={{ flex: 1 }}
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') startDownload() }}
          />
          <button onClick={startDownload}>{t('download')}</button>
        </div>
      </div>

      <div className="card">
        <h3>{t('videosHeader', { n: videos.length })}</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
          <span style={{ color: 'var(--muted)', fontSize: 12 }}>{t('threshold')}</span>
          <input type="number" step={0.05} min={0} max={1} style={{ width: 70 }}
                 value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
          <span style={{ color: 'var(--muted)', fontSize: 12 }} title={t('marginHint')}>
            {t('marginLbl')}
          </span>
          <input type="number" step={0.01} min={0} max={1} style={{ width: 70 }}
                 value={margin} onChange={e => setMargin(Number(e.target.value))} />
          <span style={{ color: 'var(--muted)', fontSize: 12 }}>
            {t('matchesAgainst')} <b>{groupName}</b>
          </span>
        </div>
        {videos.length === 0 ? (
          <div style={{ color: 'var(--muted)', fontSize: 12 }}>{t('noVideos')}</div>
        ) : (
          <table className="tracks-table">
            <thead>
              <tr>
                <th>{t('colTitle')}</th>
                <th style={{ width: 60 }}>{t('colDur')}</th>
                <th style={{ width: 60 }}>{t('colSize')}</th>
                <th style={{ width: 220 }}>{t('colQuality')}</th>
                <th style={{ width: 220 }}>{t('preview')}</th>
                <th style={{ width: 240 }}>{t('colActions')}</th>
              </tr>
            </thead>
            <tbody>
              {videosSorted.map(v => {
                const q = v.quality
                return (
                  <tr key={v.stem}>
                    <td>
                      <div>{v.title}</div>
                      <div style={{ color: 'var(--muted)', fontSize: 11 }}>
                        <a href={v.url} target="_blank" rel="noreferrer">{t('source')}</a>
                      </div>
                    </td>
                    <td>{v.duration ? `${Math.round(v.duration)}s` : '—'}</td>
                    <td>{v.size_mb != null ? `${v.size_mb} MB` : '—'}</td>
                    <td style={{ fontSize: 11 }}>
                      {q ? (
                        <>
                          <div style={{ fontSize: 16, fontWeight: 600 }}>{q.scores.composite}</div>
                          <div style={{ color: 'var(--muted)' }}>
                            {t('cutsPerSec')} {q.cuts_per_sec.toFixed(2)}/s
                            {' · '}{t('persons')} {q.median_persons}
                            {' · '}{t('heightShort')} {q.median_target_h_ratio.toFixed(2)}
                            {' · '}{t('lateralShort')} {q.target_cx_std.toFixed(2)}
                          </div>
                        </>
                      ) : (
                        <span style={{ color: 'var(--muted)' }}>{t('notProbed')}</span>
                      )}
                    </td>
                    <td>
                      {v.has_file && (
                        <video controls preload="none" src={api.videoUrl(v.stem)}
                               style={{ maxWidth: 200, maxHeight: 120 }} />
                      )}
                    </td>
                    <td>
                      <button onClick={() => startProbe(v.stem)} title={t('probeTitle')}>
                        {t('probe')}
                      </button>
                      <button onClick={() => startMatch(v.stem)} style={{ marginLeft: 6 }}>
                        {t('match')}
                      </button>
                      <button
                        className="danger"
                        style={{ marginLeft: 6 }}
                        onClick={async () => {
                          if (confirm(t('confirmDeleteVideo', { name: v.stem }))) {
                            await api.deleteVideo(v.stem); refresh()
                          }
                        }}
                      >×</button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h3>{t('clipsHeader', { group: groupName, n: groupClips.length })}</h3>
        {groupClips.length === 0 ? (
          <div style={{ color: 'var(--muted)', fontSize: 12 }}>{t('noClipsYet')}</div>
        ) : (
          <table className="tracks-table">
            <thead>
              <tr>
                <th>{t('colSong')}</th>
                <th style={{ width: 80 }}>{t('colSize')}</th>
                <th style={{ width: 320 }}>{t('preview')}</th>
                <th style={{ width: 60 }}></th>
              </tr>
            </thead>
            <tbody>
              {groupClips.map(c => (
                <tr key={c.title}>
                  <td>{c.title}</td>
                  <td>{c.size_mb} MB</td>
                  <td>
                    <video controls preload="none" src={api.clipUrl(c.group, c.song, c.title)}
                           style={{ maxWidth: 300, maxHeight: 170 }} />
                  </td>
                  <td>
                    <button
                      className="danger"
                      onClick={async () => {
                        if (confirm(t('confirmDeleteClip', { name: c.title }))) {
                          await api.deleteClip(c.group, c.song, c.title); refresh()
                        }
                      }}
                    >×</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {activeJob && (
        <div className="card">
          <h3>{t('job')}: {activeJob.kind}</h3>
          <JobLog jobId={activeJob.id} onDone={refresh} />
        </div>
      )}
    </div>
  )
}
