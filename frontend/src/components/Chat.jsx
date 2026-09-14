import { useState, useRef, useEffect } from 'react'
import { streamAsk, askAgent } from '../api.js'
import Message from './Message.jsx'

// 每个模式的独立会话：消息列表 + 多轮历史
function createSession() {
  return { messages: [], history: [] }
}

export default function Chat({ model, mode }) {
  const [sessions, setSessions] = useState(() => ({
    rag: createSession(),
    agent: createSession(),
  }))
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const idRef = useRef(0)
  const bottomRef = useRef(null)
  // Agent 会话 id：进入页面生成一次，agent 多轮追问共享同一记忆
  const threadRef = useRef(
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? `session-${crypto.randomUUID()}`
      : `session-${Math.random().toString(36).slice(2)}`
  )

  // 当前模式的会话
  const messages = sessions[mode].messages

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 更新「当前模式」的会话（闭包里的 mode 是本次发送启动时的 mode，切模式也不串）
  function updateSession(updater) {
    setSessions((prev) => ({ ...prev, [mode]: updater(prev[mode]) }))
  }

  async function send() {
    const question = input.trim()
    if (!question || loading) return
    setInput('')

    const userMsg = { id: ++idRef.current, role: 'user', content: question, sources: null }
    const aiMsg = { id: ++idRef.current, role: 'assistant', content: '', sources: null, streaming: true }
    updateSession((s) => ({ ...s, messages: [...s.messages, userMsg, aiMsg] }))
    setLoading(true)

    // 传给后端的历史，只含本次提问之前的轮次
    const history = sessions[mode].history.slice()
    updateSession((s) => ({ ...s, history: [...s.history, { role: 'user', content: question }] }))

    let answerText = ''

    try {
      if (mode === 'agent') {
        // Agent 模式：非流式，一次性拿到整段回答（带会话 id 实现多轮记忆）
        const data = await askAgent(question, threadRef.current, model)
        answerText = data.answer || '（无回答）'
        updateSession((s) => ({
          ...s,
          messages: s.messages.map((m) =>
            m.id === aiMsg.id
              ? { ...m, content: answerText, sources: data.sources || [], fallback: data.fallback || false }
              : m
          ),
        }))
      } else {
        await streamAsk(question, history, model, {
          onChunk: (chunk) => {
            answerText += chunk
            updateSession((s) => ({
              ...s,
              messages: s.messages.map((m) =>
                m.id === aiMsg.id ? { ...m, content: answerText } : m
              ),
            }))
          },
          onSources: (meta) => {
            updateSession((s) => ({
              ...s,
              messages: s.messages.map((m) =>
                m.id === aiMsg.id
                  ? { ...m, sources: meta.sources || [], fallback: meta.fallback || false }
                  : m
              ),
            }))
          },
        })
      }
    } catch (err) {
      updateSession((s) => ({
        ...s,
        messages: s.messages.map((m) =>
          m.id === aiMsg.id ? { ...m, content: m.content || `出错了：${err.message}` } : m
        ),
      }))
    } finally {
      setLoading(false)
      updateSession((s) => ({
        ...s,
        messages: s.messages.map((m) => (m.id === aiMsg.id ? { ...m, streaming: false } : m)),
      }))
      if (answerText) {
        updateSession((s) => ({ ...s, history: [...s.history, { role: 'assistant', content: answerText }] }))
      }
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="chat">
      <div className="chat-messages">
        {messages.length === 0 && <div className="chat-empty">上传文档后，向我提问吧 👇</div>}
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          rows={2}
          disabled={loading}
        />
        <button className="btn send-btn" onClick={send} disabled={loading || !input.trim()}>
          {loading ? '回答中…' : '发送'}
        </button>
      </div>
    </div>
  )
}
