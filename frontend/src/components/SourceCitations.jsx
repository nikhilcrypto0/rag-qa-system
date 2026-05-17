import { useState } from 'react'
import { ChevronDown, ChevronUp, ExternalLink, FileText } from 'lucide-react'
import styles from './SourceCitations.module.css'

export default function SourceCitations({ sources }) {
  const [open, setOpen] = useState(false)

  return (
    <div className={styles.wrapper}>
      <button
        className={styles.toggle}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        aria-controls="source-list"
      >
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        {sources.length} source{sources.length !== 1 ? 's' : ''}
      </button>
      {open && (
        <ul id="source-list" className={styles.list}>
          {sources.map((src, i) => {
            const isUrl = src.source && src.source.startsWith('http')
            return (
              <li key={i} className={styles.item}>
                <div className={styles.itemHeader}>
                  {isUrl ? <ExternalLink size={13} /> : <FileText size={13} />}
                  {isUrl ? (
                    <a href={src.source} target="_blank" rel="noopener noreferrer" className={styles.link}>
                      {src.source}
                    </a>
                  ) : (
                    <span className={styles.filename}>{src.source}</span>
                  )}
                  {src.page != null && <span className={styles.meta}>p.{src.page}</span>}
                  {src.rerank_score != null && (
                    <span className={styles.score}>{(src.rerank_score * 100).toFixed(0)}%</span>
                  )}
                </div>
                {src.excerpt && <p className={styles.excerpt}>{src.excerpt}</p>}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
