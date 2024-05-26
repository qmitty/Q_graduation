<template>
  <div class="parameter-container">
    <!-- 第一个框 -->
    <div class="parameter-box">
      <div class="box-header">一般设置</div>
      <div class="parameter-input" v-for="(param, index) in parameters1" :key="index">
        <label>{{ param.label }}</label>
        <el-tooltip class="item" effect="dark" :content="param.help" placement="top">
          <el-input
            v-model="param.value"
            :placeholder="param.help"
            :style="{ opacity: 0.6 }"
          ></el-input>
        </el-tooltip>
      </div>
    </div>

    <!-- 第二个框 -->
    <div class="parameter-box" style="position: relative">
      <div class="box-header">预训练网络设置</div>
      <div class="parameter-input" v-for="(param, index) in parameters2" :key="index">
        <label>{{ param.label }}</label>
        <el-tooltip class="item" effect="dark" :content="param.help" placement="top">
          <el-input
            v-model="param.value"
            :placeholder="param.help"
            :style="{ opacity: 0.6 }"
          ></el-input>
        </el-tooltip>
      </div>
      <el-button
        @click="sendParameters"
        type="primary"
        class="start-button"
        style="bottom: -190px; margin: 20px"
        >开始训练</el-button
      >
      <div class="start-button" style="bottom: -120px; margin-left: 18px">
        <h3 style="text-align: center">选择训练数据集</h3>
        <el-select v-model="selectedFolder" placeholder="请选择数据集" style="width: 240px">
          <el-option v-for="folder in folders" :key="folder" :value="folder">{{
            folder
          }}</el-option>
        </el-select>
      </div>
    </div>

    <!-- 第三个框 -->
    <div class="parameter-box">
      <div class="box-header">冻结训练设置</div>
      <div class="parameter-input" v-for="(param, index) in parameters3" :key="index">
        <label>{{ param.label }}</label>
        <el-tooltip class="item" effect="dark" :content="param.help" placement="top">
          <el-input
            v-model="param.value"
            :placeholder="param.help"
            :style="{ opacity: 0.6 }"
          ></el-input>
        </el-tooltip>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import axios from 'axios'

export default {
  setup() {
    const parameters1 = ref([
      { label: 'num_classes', value: '2', help: "需要的分类个数+1，默认为'2'" },
      { label: 'Init_Epoch', value: '0', help: "模型当前开始的训练世代，默认为'0'" },
      { label: 'save_period', value: '5', help: "保存模型权值的轮次数，默认为'5'" }
    ])

    const parameters2 = ref([
      { label: 'pretrained', value: 'True', help: "是否使用主干网络的预训练权重，默认为'True'" },
      { label: 'model_path', value: './', help: '模型的权值文件，默认为空' }
    ])

    const parameters3 = ref([
      { label: 'Freeze_Train', value: 'False', help: "是否进行冻结训练，默认为'False'" },
      { label: 'Freeze_Epoch', value: '0', help: "模型在解冻后的batch_size，默认为'0'" },
      { label: 'Freeze_batch_size', value: '2', help: "模型在解冻后的batch_size，默认为'2'" },
      { label: 'UnFreeze_Epoch', value: '50', help: "模型总共训练的epoch，默认为'50'" },
      { label: 'Unfreeze_batch_size', value: '2', help: "模型在解冻后的batch_size，默认为'2'" }
    ])

    const folders = ref([])
    const selectedFolder = ref('')

    const fetchFolders = () => {
      axios
        .get('http://localhost:5000/folders')
        .then((response) => {
          folders.value = response.data.folders
          console.log(folders.value)
          if (folders.value.length > 0) {
            selectedFolder.value = folders.value[0]
          }
        })
        .catch((error) => {
          console.error('There was an error fetching the folders: ', error)
        })
    }

    const sendParameters = () => {
      const params = {
        selectedFolder: selectedFolder.value
      }
      const allParameters = [...parameters1.value, ...parameters2.value, ...parameters3.value]
      allParameters.forEach((param) => {
        params[param.label] = param.value
      })

      console.log('Sending parameters:', params) // 添加这行来查看发送的参数

      axios
        .post('http://localhost:5000/train', params)
        .then((response) => {
          alert(response.data.message)
        })
        .catch((error) => {
          console.error('There was an error sending the parameters: ', error)
          alert(error.response.data.error)
        })
    }

    onMounted(() => {
      fetchFolders()
    })

    return {
      parameters1,
      parameters2,
      parameters3,
      folders,
      selectedFolder,
      fetchFolders,
      sendParameters
    }
  }
}
</script>

<style scoped>
.parameter-container {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: center;
  min-height: 100vh;
}

.box-header {
  background-color: #f0f0f0;
  text-align: center;
  padding: 10px;
}

.parameter-box {
  flex: 1;
  margin: 15px;
  border: 1px solid #ccc;
  border-radius: 5px;
  height: 430px;
  display: flex;
  flex-direction: column;
}

.parameter-input {
  margin: 10px;
}

.parameter-input label {
  display: block;
  margin-bottom: 5px;
}

.start-button {
  position: absolute;
  bottom: -60px; /* 调整按钮位置到框底部 */
  left: 60px;
  right: 0;
  margin: 0 auto; /* 设置左右外边距为自动，实现水平居中 */
  width: 240px; /* 设置按钮宽度为240px */
}
</style>
