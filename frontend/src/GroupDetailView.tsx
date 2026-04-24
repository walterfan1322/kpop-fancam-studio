import { useEffect, useState } from 'react'
import { api, type GroupDetail, type JobOut } from './api'
import { JobLog } from './JobLog'
import { VideosTab } from './VideosTab'
import { useT } from './i18n'

type Props = {
  groupName: string
  onDeleted: () => void
}

export function GroupDetailView({ groupName, onDeleted }: Props) {
  const t = useT()
  const [detail, setDetail] = useState<GroupDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [activeJob, setActiveJob] = useState<JobOut | null>(null)
  const [tab, setTab] = useState<'tracks' | 'resolve' | 'videos'>('tracks')

  const [urlText, setUrlText] = useState('')

  const refresh = async () => {
    try {
      setDetail(await api.getGroup(groupName))
      setErr(null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    refresh()
  }, [groupName])

  if (err) return <div className="empty">error: {err}</div>
  if (!detail) return <div className="empty">{t('loading')}</div>

  const updateTrack = (idx: number, patch: Partial<{ title: string; music_url: string | null }>) => {
    if (!detail) return
    const next = { ...detail, tracks: detail.tracks.slice() }
    next.tracks[idx] = { ...next.tracks[idx], ...patch }
    setDetail(next)
  }

  const removeTrack = (idx: number) => {
    if (!detail) return
    const next = { ...detail, tracks: detail.tracks.slice() }
    next.tracks.splice(idx, 1)
    setDetail(next)
  }

  const addTrack = () => {
    if (!detail) return
    setDetail({
      ...detail,
      tracks: [
        ...detail.tracks,
        { title: '', music_url: null, has_mp3: false, mp3_size_kb: null, status: null },
      ],
    })
  }

  const saveTracks = async () => {
    if (!detail) return
    const payload = detail.tracks
      .filter(t => t.title.trim())
      .map(t => ({ title: t.title.trim(), music_url: t.music_url?.trim() || null }))
    await api.replaceTracks(detail.name, payload)
    await refresh()
  }

  const startDownload = async () => {
    const job = await api.startDownload(detail.name)
    setActiveJob(job)
  }

  const startResolve = async () => {
    const urls = urlText.split(/\s+/).map(s => s.trim()).filter(Boolean)
    if (!urls.length) return
    const job = await api.startResolve(detail.name, urls)
    setActiveJob(job)
  }

  const doDelete = async () => {
    if (!confirm(t('confirmDeleteGroup', { name: detail.name }))) return
    await api.deleteGroup(detail.name)
    onDeleted()
  }

  return (
    <div>
      <h1>{detail.name}</h1>
      <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 16 }}>
        {t('artist')}: {detail.artist_name}
        {detail.mb_artist_id && <> · mbid <code>{detail.mb_artist_id}</code></>}
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="label">{t('tracks')}</div>
          <div className="value">{detail.track_count}</div>
        </div>
        <div className="metric">
          <div className="label">{t('downloaded')}</div>
          <div className="value">{detail.mp3_count}</div>
        </div>
        <div className="metric">
          <div className="label">{t('coverage')}</div>
          <div className="value">
            {detail.track_count === 0 ? '—' : `${Math.round(100 * detail.mp3_count / detail.track_count)}%`}
          </div>
        </div>
      </div>

      <div className="tab-row">
        <button className={tab === 'tracks' ? 'active' : ''} onClick={() => setTab('tracks')}>
          {t('tabTracks')}
        </button>
        <button className={tab === 'resolve' ? 'active' : ''} onClick={() => setTab('resolve')}>
          {t('tabResolve')}
        </button>
        <button className={tab === 'videos' ? 'active' : ''} onClick={() => setTab('videos')}>
          {t('tabVideos')}
        </button>
      </div>

      {tab === 'tracks' && (
        <>
          <div className="toolbar">
            <button onClick={saveTracks}>{t('saveChanges')}</button>
            <button className="secondary" onClick={addTrack}>{t('addRow')}</button>
            <button onClick={startDownload}>{t('downloadThisGroup')}</button>
            <div className="spacer" />
            <button className="danger" onClick={doDelete}>{t('deleteGroup')}</button>
          </div>

          <table className="tracks-table">
            <thead>
              <tr>
                <th style={{ width: 40 }}></th>
                <th style={{ width: 220 }}>{t('title')}</th>
                <th style={{ width: 90 }}>{t('status')}</th>
                <th>{t('musicUrl')}</th>
                <th style={{ width: 260 }}>{t('preview')}</th>
                <th style={{ width: 40 }}></th>
              </tr>
            </thead>
            <tbody>
              {detail.tracks.map((tr, i) => (
                <tr key={i}>
                  <td>
                    {tr.has_mp3 ? <span className="badge ok">{t('badgeOk')}</span>
                      : tr.status === 'no_audio' ? <span className="badge warn">{t('badgeNoAudio')}</span>
                      : tr.status === 'no_music_match' ? <span className="badge err">{t('badgeNotFound')}</span>
                      : tr.music_url ? <span className="badge warn">{t('badgePending')}</span>
                      : <span className="badge muted">—</span>}
                  </td>
                  <td>
                    <input
                      value={tr.title}
                      onChange={e => updateTrack(i, { title: e.target.value })}
                    />
                  </td>
                  <td>
                    {tr.mp3_size_kb != null ? <span style={{ color: 'var(--muted)', fontSize: 11 }}>{tr.mp3_size_kb} KB</span> : ''}
                  </td>
                  <td className="url">
                    <input
                      placeholder="https://www.tiktok.com/music/..."
                      value={tr.music_url || ''}
                      onChange={e => updateTrack(i, { music_url: e.target.value || null })}
                    />
                  </td>
                  <td>
                    {tr.has_mp3 && (
                      <audio controls preload="none" src={api.audioUrl(detail.name, tr.title)} />
                    )}
                  </td>
                  <td>
                    <button className="danger" onClick={() => removeTrack(i)}>×</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {tab === 'resolve' && (
        <div className="card">
          <h3>{t('pasteTiktokUrls')}</h3>
          <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 10 }}>
            {t('resolveHelp')}
          </div>
          <textarea
            rows={6}
            style={{ fontFamily: 'Consolas, monospace', fontSize: 12 }}
            value={urlText}
            onChange={e => setUrlText(e.target.value)}
            placeholder="https://vt.tiktok.com/xxxx/&#10;https://vt.tiktok.com/yyyy/"
          />
          <div style={{ marginTop: 10 }}>
            <button onClick={startResolve}>{t('resolveMerge')}</button>
          </div>
        </div>
      )}

      {tab === 'videos' && (
        <VideosTab groupName={detail.name} artistName={detail.artist_name} />
      )}

      {tab !== 'videos' && activeJob && (
        <div className="card">
          <h3>{t('job')}: {activeJob.kind}</h3>
          <JobLog
            jobId={activeJob.id}
            onDone={() => { refresh() }}
          />
        </div>
      )}
    </div>
  )
}
