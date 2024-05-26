<template>
  <div>
    <h2>评价指标随epoch变化</h2>
    <div class="selector-container">
      <el-select
        v-model="selectedFiles"
        placeholder="请选择文件"
        multiple
        @change="fetchData"
        style="width: 240px"
      >
        <el-option v-for="file in files" :key="file" :label="file" :value="file"></el-option>
      </el-select>
    </div>
    <div class="selector-container">
      <el-select
        v-model="selectedColumn"
        placeholder="请选择指标"
        @change="fetchData"
        style="width: 240px"
      >
        <el-option label="F-Score" value="F-Score"></el-option>
        <el-option label="MCC" value="MCC"></el-option>
        <el-option label="Loss" value="Loss"></el-option>
      </el-select>
    </div>

    <div class="chart-container" ref="chartContainer"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'
import axios from 'axios'

const chartContainer = ref(null)
let myChart = null
const selectedFiles = ref([])
const files = ref([])
const responseData = ref(new Map())
const selectedColumn = ref('F-Score') // 新增的响应式数据

const fetchFiles = async () => {
  try {
    const response = await axios.get('http://localhost:5000/files')
    files.value = response.data.files
  } catch (error) {
    console.error(error)
  }
}

const fetchData = async () => {
  for (const file of selectedFiles.value) {
    if (!responseData.value.has(file)) {
      try {
        const response = await axios.post('http://localhost:5000/hello', { filename: file })
        responseData.value.set(file, response.data)
        console.log(responseData.value)
      } catch (error) {
        console.error(error)
      }
    }
  }
  updateChartData()
}
console.log(selectedColumn.value)
const updateChartData = () => {
  if (myChart && chartContainer.value) {
    const series = []
    responseData.value.forEach((data, file) => {
      // 确保data是一个有效的数组
      if (!data || !data.length) return

      series.push({
        name: file,
        type: 'line',
        smooth: true,
        data: data.map((item) => {
          // 根据selectedColumn.value动态访问item中的属性
          const value = item[selectedColumn.value]
          return {
            value: value, // 确保这里是item中对应selectedColumn.value的值
            name: `Epoch ${item.Epoch}`
          }
        })
      })
    })

    const option = {
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: selectedFiles.value
      },
      xAxis: {
        type: 'category',
        // 确保在这里也进行了有效性检查
        data: responseData.value.get(selectedFiles.value[0])
          ? responseData.value.get(selectedFiles.value[0]).map((item) => `Epoch ${item.Epoch}`)
          : []
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}'
        }
      },
      series: series
    }

    myChart.setOption(option, true)
  }
}

onMounted(async () => {
  await fetchFiles()
  if (chartContainer.value) {
    myChart = echarts.init(chartContainer.value)
  }
})

watch(selectedFiles, fetchData, { deep: true })
</script>

<style>
.chart-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 500px;
}
.selector-container {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
}
</style>
