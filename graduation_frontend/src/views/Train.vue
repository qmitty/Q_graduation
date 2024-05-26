<template>
  <div class="upload-zip">
    <div class="upload-container">
      <el-upload
        style="width: 800px; margin-top: 30px"
        class="upload-demo"
        drag
        action="http://localhost:5000/upload-zip"
        :on-success="handleSuccess"
        :on-error="handleError"
        name="file"
        accept=".zip"
        multiple
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">拖动文件上传 <em>点击文件上传</em></div>
      </el-upload>
      <el-button type="primary">点击上传.zip文件</el-button>
      <el-button @click="unzipFiles" type="success" :loading="unzipping"> 解压文件 </el-button>
      <el-message-box v-if="message" :type="messageType">{{ message }}</el-message-box>
    </div>

    <div class="folder-list">
      <h3>文件夹列表</h3>
      <ul>
        <li v-for="folder in folders" :key="folder">{{ folder }}</li>
      </ul>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      message: '',
      messageType: 'success',
      unzipping: false,
      folders: []
    }
  },
  methods: {
    handleSuccess(response) {
      this.message = response.message
      this.messageType = 'success'
    },
    handleError(error) {
      this.message = error.response.data.error
      this.messageType = 'error'
    },
    unzipFiles() {
      this.unzipping = true
      axios
        .get('http://localhost:5000/unzip-datasets')
        .then((response) => {
          this.message = response.data.message
          this.messageType = 'success'
        })
        .catch((error) => {
          this.message = error.response.data.error
          this.messageType = 'error'
        })
        .finally(() => {
          this.unzipping = false
          location.reload()
        })
    },
    fetchFolders() {
      axios
        .get('http://localhost:5000/folders')
        .then((response) => {
          this.folders = response.data.folders
        })
        .catch((error) => {
          console.error('There was an error fetching the folders: ', error)
        })
    }
  },
  mounted() {
    this.fetchFolders()
  }
}
</script>

<style scoped>
.upload-zip {
  display: flex;
}

.upload-container {
  flex: 1;
  margin-right: 20px;
  text-align: center;
}

.folder-list {
  flex: 1;
  text-align: center; /* 添加这一行来使内容水平居中 */
}

.folder-list ul {
  list-style-type: none;
  padding: 0;
  display: inline-block; /* 使 ul 作为块级元素居中 */
  text-align: left; /* 保持列表项的文本左对齐 */
}

.folder-list li {
  margin-bottom: 5px;
}
</style>
