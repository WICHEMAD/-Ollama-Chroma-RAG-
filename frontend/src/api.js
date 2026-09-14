// 后端 API 封装。使用相对路径：开发时走 Vite 代理，生产时同源由 FastAPI 挂载。
const BASE = ''

async function parseError(res, fallback) {
  const data = await res.json().catch(() => ({}))
  return data.detail || `${fallback} (${res.status})`
}

// 获取文档统计
export async function getStats() {
  const res = await fetch(`${BASE}/documents/stats`)
  if (!res.ok) throw new Error(await parseError(res, '获取统计失败'))
  return res.json()
}

// 获取知识库文档列表
export async function getDocuments() {
  const res = await fetch(`${BASE}/documents/list`)
  if (!res.ok) throw new Error(await parseError(res, '获取文档列表失败'))
  return res.json()
}

// 获取模型列表（Ollama 已装 + 外部云端模型）
export async function getModels() {
  const res = await fetch(`${BASE}/models`)
  if (!res.ok) throw new Error(await parseError(res, '获取模型列表失败'))
  return res.json()
}

// 添加外部（云端）模型
export async function addExternalModel(spec) {
  const res = await fetch(`${BASE}/models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `添加模型失败 (${res.status})`)
  return data
}

// 删除外部模型
export async function deleteExternalModel(name) {
  const res = await fetch(`${BASE}/models/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `删除模型失败 (${res.status})`)
  return data
}

// 上传文档（multipart，字段名 file）
export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `上传失败 (${res.status})`)
  return data
}

// 流式问答（SSE）。onChunk 收到文本片段，onSources 收到来源元数据
export async function streamAsk(question, history, model, { onChunk, onSources, signal } = {}) {
  const res = await fetch(`${BASE}/qa/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history, model: model || null }),
    signal,
  })
  if (!res.ok) throw new Error(await parseError(res, '请求失败'))

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE 事件之间以空行 \n\n 分隔
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)

      for (const line of rawEvent.split('\n')) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (payload === '[DONE]') return

        let obj
        try {
          obj = JSON.parse(payload)
        } catch {
          continue
        }

        if (obj.chunk !== undefined) {
          onChunk && onChunk(obj.chunk)
        } else if (obj.type === 'sources') {
          onSources && onSources(obj)
        } else if (obj.type === 'error') {
          throw new Error(obj.detail || '生成回答失败')
        }
      }
    }
  }
}

// Agent 智能问答（非流式，一次性返回整段回答）。threadId 用于多轮记忆
export async function askAgent(question, threadId, model) {
  const res = await fetch(`${BASE}/qa/agent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, thread_id: threadId || null, model: model || null }),
  })
  if (!res.ok) throw new Error(await parseError(res, 'Agent 请求失败'))
  return res.json()
}
