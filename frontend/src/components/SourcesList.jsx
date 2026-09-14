function formatScore(score) {
  const n = Number(score)
  if (Number.isNaN(n)) return String(score)
  return n.toFixed(4)
}

export default function SourcesList({ sources }) {
  return (
    <div className="sources">
      <div className="sources-title">📎 参考来源</div>
      <ul>
        {sources.map((s, i) => (
          <li key={i}>
            <span className="source-name">{s.source || '未知来源'}</span>
            {s.page != null && <span className="source-tag">第 {s.page} 页</span>}
            {s.score != null && (
              <span className="source-tag">相关度 {formatScore(s.score)}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
