import { useEffect, useState } from 'react'
import { getStats } from '../api.js'

export default function StatsBar({ version }) {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getStats()
      .then((s) => !cancelled && setStats(s))
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [version])

  if (error) return <div className="stats-bar error">统计加载失败：{error}</div>
  if (!stats) return <div className="stats-bar">加载统计中…</div>

  return (
    <div className="stats-bar">
      <span>
        知识库：<strong>{stats.collection_name || '—'}</strong>
      </span>
      <span>
        已入库块数：<strong>{stats.document_count ?? 0}</strong>
      </span>
    </div>
  )
}
