/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        display: ["Fraunces", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        midnight: {
          950: "#0A0E1A",
          900: "#0F1424",
          800: "#161D33",
          700: "#1E2740",
          600: "#2A3555",
        },
        surface: {
          DEFAULT: "#FAF9F6",
          raised: "#FFFFFF",
        },
        accent: {
          DEFAULT: "#6C5CE7",
          bright: "#8B7FF0",
          dim: "#4A3FA0",
        },
        signal: {
          correct: "#1FAE6E",
          incorrect: "#E0485B",
          pending: "#D9A441",
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(10,14,26,0.04), 0 8px 24px -4px rgba(10,14,26,0.08)",
        cardDark: "0 1px 2px rgba(0,0,0,0.2), 0 8px 24px -4px rgba(0,0,0,0.4)",
      },
      borderRadius: {
        xl2: "1.25rem",
      },
    },
  },
  plugins: [],
};
