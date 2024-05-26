<template>
  <div class="h-6" />
  <el-menu
    :default-active="activeIndex2"
    class="el-menu-demo"
    mode="horizontal"
    background-color="#6495ED"
    text-color="#fff"
    active-text-color="#ffd04b"
    @select="handleSelect"
  >
    <el-menu-item index="1">糖网病诊断辅助系统</el-menu-item>
    <el-sub-menu index="2">
      <template #title>功能列表</template>
      <el-menu-item index="2-1">数据对比</el-menu-item>
      <el-menu-item index="2-2">数据上传</el-menu-item>
      <el-menu-item index="2-3">数据集上传</el-menu-item>
    </el-sub-menu>
    <el-menu-item index="2-6">模型训练</el-menu-item>
    <el-menu-item index="2-7">分割图像展示</el-menu-item>
  </el-menu>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

export default {
  setup() {
    const activeIndex = ref('1')
    const activeIndex2 = ref('1')
    const router = useRouter()

    const handleSelect = (key, keyPath) => {
      console.log(key, keyPath)
      if (key === '1') {
        router.push('/homeview')
      }
      if (key === '2-1') {
        router.push('/diagram')
      }
      if (key === '2-2') {
        router.push('/upload')
      }
      if (key === '2-3') {
        router.push('/train')
      }
      if (key === '2-6') {
        router.push('/trainmodel')
      }
      if (key === '2-7') {
        router.push('/segimage')
      }
    }

    return { activeIndex, activeIndex2, handleSelect }
  }
}
</script>
