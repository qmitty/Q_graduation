

import router from './router'
import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

//导入暗黑模式
import 'element-plus/theme-chalk/dark/css-vars.css'
import './styles/dark/css-vars.css';
const app = createApp(App).use(router)
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
  }
app.use(ElementPlus)
app.mount('#app')
