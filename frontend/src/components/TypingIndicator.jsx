import { GraduationCap } from 'lucide-react'
import styles from './TypingIndicator.module.css'

export default function TypingIndicator() {
  return (
    <div className={styles.row} role="status" aria-label="SEMO Assistant is thinking">
      <div className={styles.avatar}>
        <GraduationCap size={16} color="#C8102E" />
      </div>
      <div className={styles.bubble}>
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>
    </div>
  )
}
