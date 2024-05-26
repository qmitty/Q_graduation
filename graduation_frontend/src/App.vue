<script setup>
// import { RouterLink, RouterView } from 'vue-router'
import Header from './components/Header.vue'
</script>
<template>
  <div class="app-container">
    <header>
      <Header />
    </header>

    <div class="main-content">
      <RouterView />
    </div>
  </div>
</template>

<style>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh; /* 整个应用占满视口高度 */
}
.main-content {
  flex-grow: 1; /* 占据剩余的空间 */
  overflow: auto; /* 如果内容超出视口，允许滚动 */
}
</style>