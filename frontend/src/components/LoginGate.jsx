import { useState } from 'react'
import { GraduationCap, ArrowRight } from 'lucide-react'
import styles from './LoginGate.module.css'

export default function LoginGate({ onLogin }) {
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = email.trim().toLowerCase()
    if (!trimmed.endsWith('@semo.edu')) {
      setError('Please use your @semo.edu university email address.')
      return
    }
    setError('')
    onLogin(trimmed)
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.card}>
        <div className={styles.logo}>
          <GraduationCap size={32} color="#C8102E" />
        </div>
        <h1 className={styles.title}>SEMO Assistant</h1>
        <p className={styles.subtitle}>
          Your AI guide to Southeast Missouri State University
        </p>
        <form onSubmit={handleSubmit} className={styles.form}>
          <label htmlFor="email" className={styles.label}>
            University Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => { setEmail(e.target.value); setError('') }}
            placeholder="your.name@semo.edu"
            className={styles.input}
            autoComplete="email"
            required
            aria-describedby={error ? 'email-error' : undefined}
          />
          {error && (
            <p id="email-error" className={styles.error} role="alert">
              {error}
            </p>
          )}
          <button type="submit" className={styles.btn}>
            Enter SEMO Assistant
            <ArrowRight size={18} />
          </button>
        </form>
        <p className={styles.note}>
          Access restricted to Southeast Missouri State University students and staff.
        </p>
      </div>
    </div>
  )
}
