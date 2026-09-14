import { useRef, useState } from 'react'
import { uploadDocument } from '../api.js'

export default function UploadPanel({ onUploaded }) {
  const inputRef = useRef(null)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState(null) // { type: 'ok' | 'err', text }

  async function handleFile(e) {
    const file = e.target.files[0]
    if (!file) return
    setUploading(true)
    setMessage(null)
    try {
      const data = await uploadDocument(file)
      if (data.status === 'duplicate') {
        setMessage({ type: 'err', text: `「${data.filename}」已存在，跳过入库` })
      } else {
        setMessage({ type: 'ok', text: `「${data.filename}」上传成功，入库 ${data.stored_count} 块` })
      }
      onUploaded && onUploaded()
    } catch (err) {
      setMessage({ type: 'err', text: err.message })
    } finally {
      setUploading(false)
      e.target.value = '' // 允许重复选择同一文件
    }
  }

  return (
    <div className="panel upload-panel">
      <h2>文档上传</h2>
      <p className="hint">支持 PDF / TXT，上传后自动向量化入库</p>
      <button className="btn" disabled={uploading} onClick={() => inputRef.current?.click()}>
        {uploading ? '上传中…' : '选择文件上传'}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,application/pdf,text/plain"
        hidden
        onChange={handleFile}
      />
      {message && <p className={`upload-msg ${message.type}`}>{message.text}</p>}
    </div>
  )
}
