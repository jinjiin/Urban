// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import axios from 'axios'
import router from './router'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import echarts from 'echarts'
import BaiduMap from 'vue-baidu-map'
import globalVariable from '@/api/global_variable.js'
import globalConfig from '@/api/global_config.js'
import globalFunction from '@/api/interface.js'

Vue.prototype.$axios = axios;
Vue.prototype.$echarts = echarts;
Vue.use(ElementUI);
Vue.config.productionTip = false;

// Replace by yours
Vue.use(BaiduMap, {
  ak: "S9ES8ITKOUuZYiR9DnMoYr1BIZ1DPvdp"
})
// Global Variables
Vue.prototype.global_ = globalVariable;
Vue.prototype.config_ = globalConfig;
Vue.use(globalFunction);

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
