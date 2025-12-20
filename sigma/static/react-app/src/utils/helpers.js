export function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

export function formatToolResult(content) {
  try {
    const parsed = JSON.parse(content)
    return { parsed, isJson: true }
  } catch {
    return { content, isJson: false }
  }
}
