import { useState, useEffect } from 'react'
import { getModels, addExternalModel, deleteExternalModel } from '../api.js'

export default function ModelSelector({ value, onChange, mode, onModeChange }) {
  const [external, setExternal] = useState([]) // 云端外部模型（后端持久化）
  const [ext, setExt] = useState({ model: '', base_url: '', api_key: '' })
  const [error, setError] = useState('')

  async function refresh() {
    try {
      const data = await getModels()
      setExternal((data.models || []).map((m) => m.name))
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  async function addExternal() {
    if (!ext.model.trim()) {
      setError('模型 ID 不能为空')
      return
    }
    try {
      const name = ext.model.trim()
      await addExternalModel({
        name,
        model: name,
        provider: 'openai',
        base_url: ext.base_url.trim() || null,
        api_key: ext.api_key.trim() || null,
      })
      setExt({ model: '', base_url: '', api_key: '' })
      setError('')
      await refresh()
      onChange(name)
    } catch (e) {
      setError(e.message)
    }
  }

  async function removeExternal(name) {
    try {
      await deleteExternalModel(name)
      setError('')
      if (value === name) onChange('')
      await refresh()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="panel model-selector">
      <h2>🧠 对话模型</h2>
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode !== 'agent' ? 'active' : ''}`}
          onClick={() => onModeChange('rag')}
        >
          普通问答
        </button>
        <button
          className={`mode-btn ${mode === 'agent' ? 'active' : ''}`}
          onClick={() => onModeChange('agent')}
        >
          Agent 智能
        </button>
      </div>
      <select value={value || ''} onChange={(e) => onChange(e.target.value || '')}>
        <option value="" disabled>
          请选择云端模型
        </option>
        {external.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>

      {/* 云端模型表单 */}
      <div className="model-external">
        <input
          value={ext.model}
          onChange={(e) => setExt({ ...ext, model: e.target.value })}
          placeholder="模型 ID，如 qwen-plus"
        />
        <input
          value={ext.base_url}
          onChange={(e) => setExt({ ...ext, base_url: e.target.value })}
          placeholder="API 地址 base_url（可选）"
        />
        <input
          type="password"
          value={ext.api_key}
          onChange={(e) => setExt({ ...ext, api_key: e.target.value })}
          placeholder="API Key（sk-...）"
        />
        <button className="btn" onClick={addExternal} disabled={!ext.model.trim()}>
          添加云端模型
        </button>
      </div>

      {/* 已添加的云端模型列表（可删除） */}
      {external.length > 0 && (
        <div className="external-list">
          {external.map((name) => (
            <div className="external-item" key={name}>
              <span>{name}</span>
              <button className="external-del" onClick={() => removeExternal(name)} title="删除">
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {error && <div className="upload-msg err">{error}</div>}
    </div>
  )
}
