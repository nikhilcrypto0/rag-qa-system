import { useState, useRef, useEffect, useCallback } from 'react'
import { LogOut, Send, Zap, ZapOff, GraduationCap } from 'lucide-react'
import MessageBubble from './MessageBubble.jsx'
import TypingIndicator from './TypingIndicator.jsx'
import styles from './ChatWindow.module.css'

export default function ChatWindow({ userEmail, onLogout, apiBase }) {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: `Hi! I'm the SEMO Assistant. Ask me anything about Southeast Missouri State University — admissions, courses, financial aid, campus life, and more.`,
      sources: [],
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [streaming, setStreaming] = useState(true)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const sendMessage = useCallback(async () => {
    const question = input.trim()
    if (!question || isLoading) return

    setInput('')
    setError('')
    const userMsg = { id: Date.now().toString(), role: 'user', content: question, sources: [] }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    const headers = {
      'Content-Type': 'application/json',
      'X-User-Email': userEmail,
    }

    if (streaming) {
      // SSE streaming
      const assistantId = (Date.now() + 1).toString()
      setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '', sources: [], streaming: true }])

      try {
        const controller = new AbortController()
        abortRef.current = controller
        const res = await fetch(`${apiBase}/query/stream`, {
          method: 'POST',
          headers,
          body: JSON.stringify({ question }),
          signal: controller.signal,
        })

        if (!res.ok) throw new Error(`Server error ${res.status}`)

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let fullText = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const raw = line.slice(6)
              if (raw === '[DONE]') break
              try {
                const token = JSON.parse(raw)
                fullText += token
              } catch {
                fullText += raw
              }
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, content: fullText } : m
              ))
            }
          }
        }
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, streaming: false } : m
        ))
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError('Something went wrong. Please try again.')
          setMessages(prev => prev.filter(m => m.id !== assistantId))
        }
      }
    } else {
      // Non-streaming
      try {
        const res = await fetch(`${apiBase}/query`, {
          method: 'POST',
          headers,
          body: JSON.stringify({ question }),
        })
        if (!res.ok) throw new Error(`Server error ${res.status}`)
        const data = await res.json()
        setMessages(prev => [...prev, {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
          latency: data.total_latency_ms,
        }])
      } catch (err) {
        setError('Something went wrong. Please try again.')
      }
    }

    setIsLoading(false)
    inputRef.current?.focus()
  }, [input, isLoading, streaming, userEmail, apiBase])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className={styles.layout}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.headerIcon}>
            <GraduationCap size={20} color="#C8102E" />
          </div>
          <div>
            <span className={styles.headerTitle}>SEMO Assistant</span>
            <span className={styles.headerSub}>Southeast Missouri State University</span>
          </div>
        </div>
        <div className={styles.headerRight}>
          <button
            className={`${styles.streamToggle} ${streaming ? styles.streamOn : ''}`}
            onClick={() => setStreaming(s => !s)}
            title={streaming ? 'Streaming on' : 'Streaming off'}
            aria-label={streaming ? 'Disable streaming' : 'Enable streaming'}
          >
            {streaming ? <Zap size={15} /> : <ZapOff size={15} />}
            {streaming ? 'Stream' : 'Static'}
          </button>
          <span className={styles.emailBadge}>{userEmail}</span>
          <button className={styles.logoutBtn} onClick={onLogout} aria-label="Logout">
            <LogOut size={16} />
          </button>
        </div>
      </header>

      {/* Messages */}
      <main className={styles.messages} role="log" aria-live="polite" aria-label="Chat messages">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isLoading && !streaming && <TypingIndicator />}
        {error && (
          <div className={styles.errorBanner} role="alert">
            {error}
            <button onClick={() => setError('')} className={styles.errorDismiss}>Dismiss</button>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className={styles.inputArea}>
        <div className={styles.inputWrapper}>
          <textarea
            ref={inputRef}
            className={styles.input}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about SEMO admissions, courses, financial aid..."
            rows={1}
            aria-label="Your message"
            disabled={isLoading}
          />
          <button
            className={styles.sendBtn}
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
          >
            <Send size={18} />
          </button>
        </div>
        <p className={styles.hint}>Press Enter to send · Shift+Enter for new line</p>
      </footer>
    </div>
  )
}
