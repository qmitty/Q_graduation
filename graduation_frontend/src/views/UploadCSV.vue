<template>
  <div class="container">
    <el-upload
      style="width: 600px; margin-top: 30px"
      class="upload-demo"
      drag
      action="http://localhost:5000/upload"
      :on-success="handleSuccess"
      :on-error="handleError"
      multiple
    >
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">拖动文件上传 <em>点击文件上传</em></div>
      <template #tip>
        <div class="el-upload__tip">请上传评估结果文件</div>
      </template>
    </el-upload>
  </div>

  <div class="myclass">
    <el-carousel indicator-position="outside">
      <el-carousel-item v-for="item in 4" :key="item">
        <img src="/images/image.jpg" alt="image" style="width: 100%" />
      </el-carousel-item>
    </el-carousel>
  </div>
</template>

<script>
export default {
  methods: {
    handleSuccess(response, file, fileList) {
      console.log('文件上传成功:', response)
    },
    handleError(err, file, fileList) {
      console.error('文件上传失败:', err)
    }
  }
}
</script>

<style scoped>
.container {
  display: flex; /* 启用Flex布局 */
  justify-content: center;

  /* align-items: center; 垂直居中 */
  /* height: 100vh; 容器高度设置为视窗的100% */
}
.el-carousel__item h3 {
  display: flex;
  color: #475669;
  opacity: 0.75;
  line-height: 300px;
  margin: 0;
}
.myclass {
  position: absolute;
  left: 60px;
  right: 0;
  margin: 0 auto; /* 设置左右外边距为自动，实现水平居中 */

  margin-top: 70px;
  width: 900px;
}

.el-carousel__item:nth-child(2n) {
  background-color: #99a9bf;
}

.el-carousel__item:nth-child(2n + 1) {
  background-color: #d3dce6;
}
</style>
