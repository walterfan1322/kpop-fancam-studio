import { useCallback, useContext, useEffect, useState } from 'react'
import { api, type GroupSummary, type JobOut } from './api'
import { GroupDetailView } from './GroupDetailView'
import { JobLog } from './JobLog'
import { QuickMode } from './QuickMode'
import { LangContext, useLangState, useT } from './i18n'

type Mode = 'simple' | 'admin'

function TopBar({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  const t = useT()
  const { lang, setLang } = useContext(LangContext)
  return (
    <div className="topbar">
      <div className="mode-toggle">
        <button className={mode === 'simple' ? 'active' : ''} onClick={() => setMode('simple')}>
          {t('modeSimple')}
        </button>
        <button className={mode === 'admin' ? 'active' : ''} onClick={() => setMode('admin')}>
          {t('modeAdmin')}
        </button>
      </div>
      <div className="lang-toggle">
        <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>
          {t('langEn')}
        </button>
        <button className={lang === 'zh' ? 'active' : ''} onClick={() => setLang('zh')}>
          {t('langZh')}
        </button>
      </div>
    </div>
  )
}

function AdminShell() {
  const t = useT()
  const [groups, setGroups] = useState<GroupSummary[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [newGroup, setNewGroup] = useState('')
  const [limit, setLimit] = useState(40)
  const [fetchJob, setFetchJob] = useState<JobOut | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const list = await api.listGroups()
      setGroups(list)
      setErr(null)
      if (!selected && list.length > 0) setSelected(list[0].name)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [selected])

  useEffect(() => { refresh() }, [refresh])

  const startFetch = async () => {
    const name = newGroup.trim()
    if (!name) return
    const job = await api.startFetch(name, limit)
    setFetchJob(job)
    setNewGroup('')
  }

  const onDeleted = async () => {
    setSelected(null)
    await refresh()
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 style={{ margin: 0 }}>{t('appTitle')}</h1>

        <h2>{t('groups')}</h2>
        {err && <div style={{ color: 'var(--err)', fontSize: 12, marginBottom: 8 }}>{err}</div>}
        {groups === null ? (
          <div style={{ color: 'var(--muted)' }}>{t('loading')}</div>
        ) : groups.length === 0 ? (
          <div style={{ color: 'var(--muted)', fontSize: 12 }}>{t('noGroupsYet')}</div>
        ) : (
          groups.map(g => (
            <div
              key={g.name}
              className={`group-item ${selected === g.name ? 'active' : ''}`}
              onClick={() => setSelected(g.name)}
            >
              <div className="name">{g.name}</div>
              <div className="meta">
                {g.mp3_count}/{g.track_count} {t('mp3sCount')}
              </div>
            </div>
          ))
        )}

        <h2 style={{ marginTop: 26 }}>{t('addGroup')}</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <input
            placeholder={t('groupPlaceholder')}
            value={newGroup}
            onChange={e => setNewGroup(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') startFetch() }}
          />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ color: 'var(--muted)', fontSize: 12 }}>{t('limit')}</span>
            <input
              type="number"
              style={{ width: 80 }}
              value={limit}
              min={0}
              max={200}
              onChange={e => setLimit(Number(e.target.value))}
            />
          </div>
          <button onClick={startFetch}>{t('fetchMB')}</button>
        </div>

        {fetchJob && (
          <div style={{ marginTop: 16 }}>
            <JobLog jobId={fetchJob.id} onDone={refresh} />
          </div>
        )}
      </aside>

      <main className="main">
        {selected ? (
          <GroupDetailView key={selected} groupName={selected} onDeleted={onDeleted} />
        ) : (
          <div className="empty">{t('selectGroupPrompt')}</div>
        )}
      </main>
    </div>
  )
}

function AppBody() {
  const [mode, setModeState] = useState<Mode>(() =>
    (localStorage.getItem('mode') as Mode) || 'simple'
  )
  const setMode = (m: Mode) => {
    localStorage.setItem('mode', m)
    setModeState(m)
  }
  return (
    <>
      <TopBar mode={mode} setMode={setMode} />
      {mode === 'simple' ? <QuickMode /> : <AdminShell />}
    </>
  )
}

export default function App() {
  const state = useLangState()
  return (
    <LangContext.Provider value={state}>
      <AppBody />
    </LangContext.Provider>
  )
}
