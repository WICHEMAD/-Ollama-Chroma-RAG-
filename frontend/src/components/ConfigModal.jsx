import { useState } from 'react'
import { addExternalModel } from '../api.js'

// 首屏 / 修改配置 共用的云端模型配置弹窗。
// onAdded(name)：添加成功后回调（用于把该模型设为当前模型）；
// onDone()：点「完成」关闭弹窗（调用方负责持久化标记）。
export default function ConfigModal({ onAdded, onDone }) {
  const [form, setForm] = useState({ model: '', base_url: '', api_key: '' })
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  async function add() {
    if (!form.model.trim()) {
      setError('模型 ID 不能为空')
      return
    }
    setSaving(true)
    try {
      const name = form.model.trim()
      await addExternalModel({
        name,
        model: name,
        provider: 'openai',
        base_url: form.base_url.trim() || null,
        api_key: form.api_key.trim() || null,
      })
      setError('')
      onAdded(name)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h2>⚙️ 配置云端模型</h2>
        <p className="hint">对话生成已全切云端，本地只保留改写与嵌入。填入云端模型信息，配置后会自动记住，下次无需重复填写。</p>

        <input
          value={form.model}
          onChange={(e) => setForm({ ...form, model: e.target.value })}
          placeholder="模型 ID，如 qwen-plus / deepseek-chat"
        />
        <input
          value={form.base_url}
          onChange={(e) => setForm({ ...form, base_url: e.target.value })}
          placeholder="API 地址 base_url（可选）"
        />
        <input
          type="password"
          value={form.api_key}
          onChange={(e) => setForm({ ...form, api_key: e.target.value })}
          placeholder="API Key（sk-...）"
        />

        {error && <div className="upload-msg err">{error}</div>}

        <div className="modal-actions">
          <button className="btn" onClick={add} disabled={saving || !form.model.trim()}>
            {saving ? '添加中…' : '添加并启用'}
          </button>
          <button className="btn btn-ghost" onClick={onDone} disabled={saving}>
            完成
          </button>
        </div>
      </div>
    </div>
  )
}
