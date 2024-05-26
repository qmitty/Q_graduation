import { ref } from 'vue';
import axios from 'axios';
import { API_BASE_URL } from '@/api/apiConfig'; // 根据实际路径导入API_BASE_URL

export const sharedData = ref(null);

export const fetchAndStoreData = async () => {
  const token = localStorage.getItem('access_token');
  try {
    const response = await axios.get(`${API_BASE_URL}/sql/training-info/`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    sharedData.value = response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
  }
};

// 获取 sharedData 的方法
export const getSharedData = () => {
  return sharedData.value;
};
// 在模块加载时立即调用 fetchAndStoreData
fetchAndStoreData();