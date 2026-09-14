import { useState, useEffect, useCallback } from 'react'
import { getDocuments } from '../api.js'

// 把字节数格式化成可读大小：B / KB / MB
function formatSize(bytes) {
  if (bytes == null || isNaN(bytes)) return '未知'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

// 把上传时间（时间戳秒，或后端已格式化的字符串）转成可读文本
function formatTime(value) {
  if (value == null || value === '') return '—'
  if (typeof value === 'number') {
    const d = new Date(value * 1000)
    if (isNaN(d.getTime())) return '—'
    const pad = (n) => String(n).padStart(2, '0')
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
  }
  return String(value)
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await getDocuments()
      setDocs(data.documents || [])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return (
    <div className="panel documents-page">
      <div className="documents-header">
        <h2>📄 知识库文档</h2>
        <button className="btn btn-ghost" onClick={refresh} disabled={loading}>
          {loading ? '加载中…' : '刷新'}
        </button>
      </div>

      {error && <p className="upload-msg err">{error}</p>}

      {!loading && !error && docs.length === 0 && (
        <p className="documents-empty">暂无文档，请先在左侧上传 PDF / TXT。</p>
      )}

      {docs.length > 0 && (
        <div className="doc-list">
          <div className="doc-list-title">共 {docs.length} 份文档</div>
          <ul className="doc-list-items">
            {docs.map((d, idx) => (
              <li key={d.filename + '-' + idx} className="doc-item">
                <div className="doc-item-name">{d.filename}</div>
                <div className="doc-item-meta">
                  <span>{formatSize(d.size)}</span>
                  <span>{d.chunks ?? 0} 块</span>
                  <span>{formatTime(d.uploaded_at)}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
