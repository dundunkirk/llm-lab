import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  base: '/llm-lab/',
  title: 'llm-lab',
  description: 'LLM 原理与工程实践的个人学习记录',
  lang: 'zh-CN',

  markdown: {
    config: (md) => {
      md.use(mathjax3)
    },
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '路线图', link: '/roadmap/roadmap' },
      { text: 'GitHub', link: 'https://github.com/dundunkirk/llm-lab' },
    ],

    sidebar: [
      {
        text: '基础笔记',
        items: [
          { text: 'Micrograd', link: '/basis/micrograd' },
          { text: 'XGBoost', link: '/basis/xgboost' },
        ],
      },
      {
        text: '路线图',
        items: [
          { text: '学习路线', link: '/roadmap/roadmap' },
          { text: '12 周执行计划', link: '/roadmap/study_plan' },
          { text: '学习资源', link: '/roadmap/llm_link' },
          { text: 'modern_ai_for_beginners', link: '/roadmap/modern_ai_for_beginners' },
          { text: '如何学习大模型', link: '/roadmap/how_to_learn_llm' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/dundunkirk/llm-lab' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'LLM 原理与工程实践的个人学习记录',
    },
  },
})
