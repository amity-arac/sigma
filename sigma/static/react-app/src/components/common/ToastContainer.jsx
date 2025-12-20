import { useToast } from '../../context/ToastContext'
import './ToastContainer.css'

function ToastContainer() {
  const { toasts } = useToast()

  return (
    <div className="toast-container">
      {toasts.map(toast => (
        <div key={toast.id} className={`toast ${toast.type}`}>
          {toast.message}
        </div>
      ))}
    </div>
  )
}

export default ToastContainer
