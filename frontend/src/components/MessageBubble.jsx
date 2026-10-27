import ReactMarkdown from 'react-markdown'
import { User, GraduationCap, ThumbsUp, ThumbsDown } from 'lucide-react'
import SourceCitations from './SourceCitations.jsx'
import styles from './MessageBubble.module.css'

export default function MessageBubble({ message, onFeedback }) {
  const isUser = message.role === 'user'

  return (
    <div className={`${styles.row} ${isUser ? styles.userRow : styles.assistantRow}`}>
      {!isUser && (
        <div className={styles.avatar} aria-hidden="true">
          <GraduationCap size={16} color="#C8102E" />
        </div>
      )}
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
        {isUser ? (
          <p className={styles.content}>{message.content}</p>
        ) : (
          <div className={`${styles.content} ${message.streaming ? styles.streaming : ''}`}>
            <ReactMarkdown>{message.content || (message.streaming ? '' : '…')}</ReactMarkdown>
            {message.streaming && <span className={styles.cursor} aria-hidden="true" />}
          </div>
        )}
        {!isUser && message.sources && message.sources.length > 0 && (
          <SourceCitations sources={message.sources} />
        )}
        {message.degraded && (
          <span className={styles.degraded}>Keyword search was offline, so this answer may be less precise.</span>
        )}
        {!isUser && message.responseId && !message.streaming && (
          <div className={styles.feedback}>
            {message.rating ? (
              <span className={styles.feedbackThanks}>Thanks for the feedback</span>
            ) : (
              <>
                <button
                  className={styles.feedbackBtn}
                  onClick={() => onFeedback(message.id, message.responseId, 'up')}
                  aria-label="Helpful answer"
                >
                  <ThumbsUp size={14} />
                </button>
                <button
                  className={styles.feedbackBtn}
                  onClick={() => onFeedback(message.id, message.responseId, 'down')}
                  aria-label="Unhelpful answer"
                >
                  <ThumbsDown size={14} />
                </button>
              </>
            )}
          </div>
        )}
        {message.latency && (
          <span className={styles.latency}>{Math.round(message.latency)}ms</span>
        )}
      </div>
      {isUser && (
        <div className={`${styles.avatar} ${styles.userAvatar}`} aria-hidden="true">
          <User size={16} color="#94A3B8" />
        </div>
      )}
    </div>
  )
}
