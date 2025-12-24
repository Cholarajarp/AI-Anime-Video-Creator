"""
Theme Toggle Component
======================
Dark/light theme switching functionality.
"""

import gradio as gr


def create_theme_toggle() -> gr.Button:
    """
    Create the theme toggle button.

    Returns:
        Theme toggle button component
    """
    return gr.Button(
        "🌙",
        variant="secondary",
        size="sm",
        elem_classes=["theme-toggle-btn"],
        elem_id="theme-toggle"
    )


def get_theme_css() -> str:
    """
    Get CSS for theme switching.

    Returns:
        CSS string with dark/light theme variables
    """
    return """
    :root {
        /* Light Theme (Default) */
        --bg-primary: #ffffff;
        --bg-secondary: #f3f4f6;
        --bg-tertiary: #e5e7eb;
        --text-primary: #111827;
        --text-secondary: #4b5563;
        --text-muted: #9ca3af;
        --border-color: #e5e7eb;
        --accent-color: #6366f1;
        --accent-hover: #4f46e5;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    [data-theme="dark"] {
        /* Dark Theme */
        --bg-primary: #111827;
        --bg-secondary: #1f2937;
        --bg-tertiary: #374151;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --text-muted: #6b7280;
        --border-color: #374151;
        --accent-color: #818cf8;
        --accent-hover: #6366f1;
        --success-color: #34d399;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    """


def get_theme_toggle_js() -> str:
    """
    Get JavaScript for theme toggle functionality.

    Returns:
        JavaScript code string
    """
    return """
    <script>
    function toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Update button icon
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.textContent = newTheme === 'dark' ? '☀️' : '🌙';
        }
    }
    
    // Initialize theme on page load
    (function() {
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const theme = savedTheme || (prefersDark ? 'dark' : 'light');
        
        document.documentElement.setAttribute('data-theme', theme);
        
        // Update button after Gradio loads
        setTimeout(function() {
            const btn = document.getElementById('theme-toggle');
            if (btn) {
                btn.textContent = theme === 'dark' ? '☀️' : '🌙';
                btn.onclick = toggleTheme;
            }
        }, 100);
    })();
    </script>
    """


def get_theme_toggle_handler():
    """
    Get the Gradio event handler for theme toggling.

    Note: This uses JavaScript injection since Gradio doesn't have
    native theme switching.
    """
    js_code = """
    () => {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        return newTheme === 'dark' ? '☀️' : '🌙';
    }
    """
    return js_code

