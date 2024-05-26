<template>
    <div class="demo-image__lazy">
      <el-image v-for="url in urls" :key="url" :src="url" lazy />
    </div>
  </template>
  
  <script setup>
  const start = 14;
  const end = 50; // 假设结束值为20，可以根据需要调整
  const urls = [];
  
  for (let i = start; i <= end; i++) {
    urls.push(`http://localhost:5000/image/visualization_${i}.png`);
  }
  </script>
  
  <style scoped>
  .demo-image__lazy {
    height: 800px;
    overflow-y: auto;
  }
  
.demo-image__lazy .el-image {
display: block;
  width: 50%; /* 设置宽度为50% */
  height: auto; /* 自动调整高度 */
  margin-bottom: 10px; /* 删除图像之间的空白 */
  margin: 10px auto; /* 让图像水平居中 */

}

  .demo-image__lazy .el-image:last-child {
    margin-bottom: 0;
  }
  </style>
  