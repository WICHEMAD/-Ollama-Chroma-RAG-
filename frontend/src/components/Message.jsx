import SourcesList from './SourcesList.jsx'

export default function Message({ message }) {
  const isUser = message.role === 'user'
  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-role">{isUser ? '你' : 'AI'}</div>
      <div className="message-bubble">
        <div className="message-content">
          {message.content}
          {!isUser && message.streaming && <span className="cursor">▍</span>}
        </div>
        {!isUser && message.fallback && (
          <div className="fallback-notice">⚠️ 云端模型调用失败，已回退本地模型</div>
        )}
        {!isUser && message.sources && message.sources.length > 0 && (
          <SourcesList sources={message.sources} />
        )}
      </div>
    </div>
  )
}
