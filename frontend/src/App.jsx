import { useState, useEffect } from 'react'
import LoginGate from './components/LoginGate.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [userEmail, setUserEmail] = useState(() => localStorage.getItem('semo_email') || '')

  const handleLogin = (email) => {
    localStorage.setItem('semo_email', email)
    setUserEmail(email)
  }

  const handleLogout = () => {
    localStorage.removeItem('semo_email')
    setUserEmail('')
  }

  if (!userEmail) {
    return <LoginGate onLogin={handleLogin} />
  }

  return <ChatWindow userEmail={userEmail} onLogout={handleLogout} apiBase={API_BASE} />
}
