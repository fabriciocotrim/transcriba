// Gera css/site.css a partir das classes usadas no index.html. Rodar depois de mudar classes:
//   npx tailwindcss@3.4.17 -i css/entrada.css -o css/site.css --minify
module.exports = {
  content: ['./index.html'],
  theme: {
    extend: {
      colors: {
        paper: '#F2EFE9',
        paperDark: '#E6E2D8',
        paperRow: '#E5E1D6',
        ink: '#141413',
        inkMuted: '#66645E',
        editorialRed: '#B82921',
        editorialRedHover: '#961E17',
      },
      fontFamily: {
        serif: ['"Newsreader"', 'Georgia', 'serif'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
};
