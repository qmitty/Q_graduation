<template>
  <div class="home-view">
    <div class="title-section text-center">
      <h1>欢迎使用糖网病智能诊断系统</h1>
    </div>
    <div class="tech-section">
      <el-row :gutter="20">
        <el-col :span="6" class="tech-item" v-for="item in techItems" :key="item.name">
          <el-card>
            <div class="text-center">
              <a :href="item.url" target="_blank">
                <img :src="item.icon" alt="tech-icon" class="tech-icon" />
              </a>
              <p>{{ item.name }}</p>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      techItems: [
        { name: 'Vue', icon: 'public/images/vue.png', url: 'https://vuejs.org/' },
        {
          name: 'HTML',
          icon: 'public/images/html.png',
          url: 'https://developer.mozilla.org/en-US/docs/Web/HTML'
        },
        { name: 'Node', icon: 'public/images/node.png', url: 'https://nodejs.org/' },
        {
          name: 'JavaScript',
          icon: 'public/images/js.png',
          url: 'https://developer.mozilla.org/en-US/docs/Web/JavaScript'
        }
      ]
    }
  }
}
</script>

<style scoped>
.home-view {
  margin: 88px;
}

.title-section h1 {
  font-size: 30px;
  color: #333;
  margin-bottom: 98px;
}

.tech-section .tech-item .tech-icon {
  width: 200px;
  height: auto;
  margin-bottom: 10px;
}

.text-center {
  text-align: center;
}
</style>
