<template>
  <div style="height:100%">
    <el-row>
      <el-select v-model="method_value" placeholder="Select Method">
        <el-option v-for="item in method_options" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <!--
      <el-select v-model="poi_value" placeholder="Select ID">
        <el-option v-for="item in poi_options" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-select v-model="attr_value" collapse-tags style="margin-left: 20px;" placeholder="Select Attr">
        <el-option v-for="item in attr_options" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      -->
      &nbsp
      <el-button type="success" @click="run">Predict</el-button>
    </el-row>
    <br />
    <!--
    <el-row type="flex">
      <el-col :span="2" style="float:left">
        <el-input v-model="params.lr" placeholder="Leanring Rate"></el-input>
      </el-col>
      &nbsp
      <el-col :span="2">
        <el-input v-model="params.history" placeholder="History"></el-input>
      </el-col>
      &nbsp
      <el-col :span="2">
        <el-input v-model="params.future" placeholder="Future"></el-input>
      </el-col>
    </el-row>
    -->
    <el-row style="height:80%;width: 100%">
      <div ref="chart" class="map-class">
      </div>
    </el-row>
  </div>
</template>
<script>
require('echarts/extension/bmap/bmap')
export default {
  name: 'TimeSeriesPredict',
  data() {
    return {
      params: {
        lr: null,
        history: null,
        future: null
      },
      poi_options: [],
      attr_options: [],
      method_options: [{ value: "LSTM", label: "LSTM" }, { value: "GEOMAN", label: "GEOMAN" }, { value: "Attention Model", label: "Attention Model" }],
      poi_value: '',
      attr_value: '',
      method_value: '',
      Chart: null,
    }
  },
  mounted: function() {
    var promise = this.GetIdList(this.global_.data_name);
    promise.then((response) => {
      var data = response.data;
      this.poi_options = []
      for (var i = 0; i < data.length; ++i) {
        if (data[i].length > 0) {
          this.poi_options.push({ value: data[i], label: data[i] });
        }
      }
    });
    promise = this.GetAttrList(this.global_.data_name);
    promise.then((response) => {
      var data = response.data;
      this.attr_options = []
      for (var i = 0; i < data.length; ++i) {
        this.attr_options.push({ value: data[i], label: data[i] });
      }
    });
    this.Chart = this.$echarts.init(this.$refs.chart);
    window.addEventListener('resize', () => {
      this.Chart.resize();
    })
  },
  methods: {
    run() {
      var promise = this.PredictTimeSeries(this.global_.data_name, this.attr_value, this.poi_value);
      promise.then((response) => {
        var data = response.data;
        var timeline = [];
        var a1 = [];
        var a2 = [];
        var a3 = [];
        var a4 = [];
        var a5 = [];
        var a6 = [];
        console.log(data)
        for (var i = 0; i < data[0].length; ++i) {
          timeline.push(data[0][i]);
          a1.push(data[1][i]);
          a2.push(data[2][i]);
          a3.push(data[3][i]);
          a4.push(data[4][i]);
          a5.push(data[5][i]);
          a6.push(data[6][i]);
        }
        this.draw(timeline, a1, a2, a3, a4, a5, a6);
      });
    },
    draw(timeline, a1, a2, a3, a4, a5, a6) {
      var option = {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            lineStyle: {
              color: '#ADADADFF'
            }
          }
        },
        legend: {
          icon: 'rect',
          itemWidth: 14,
          itemHeight: 5,
          itemGap: 13,
          data: ["AQI", "no2", "o3", "pm10", "pm25", "so2"],
          right: '4%',
          textStyle: {
            fontSize: 15,
          }
        },
        grid: [
          {left: '2%', right: '67%', top: '9%', bottom: '47%', containLabel: true},
          {left: '34%', right: '35%', top: '9%', bottom: '47%', containLabel: true},
          {left: '66%', right: '3%', top: '9%', bottom: '47%', containLabel: true},
          {left: '2%', right: '67%', top: '55%', bottom: '1%', containLabel: true},
          {left: '34%', right: '35%', top: '55%', bottom: '1%', containLabel: true},
          {left: '66%', right: '3%', top: '55%', bottom: '1%', containLabel: true}
        ],
        xAxis: [
          {type: 'category',
          boundaryGap: false,
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          textStyle: {
            fontSize: 15,
            color: "#000000FF"
          },
          data: timeline,
          gridIndex: 0},
          {type: 'category',
          boundaryGap: false,
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          textStyle: {
            fontSize: 15,
            color: "#000000FF"
          },
          data: timeline,
          gridIndex: 1},
          {type: 'category',
          boundaryGap: false,
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          textStyle: {
            fontSize: 15,
            color: "#000000FF"
          },
          data: timeline,
          gridIndex: 2},
          {type: 'category',
          boundaryGap: false,
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          textStyle: {
            fontSize: 15,
            color: "#000000FF"
          },
          data: timeline,
          gridIndex: 3},
          {type: 'category',
          boundaryGap: false,
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          textStyle: {
            fontSize: 15,
            color: "#000000FF"
          },
          data: timeline,
          gridIndex: 4},
          {type: 'category',
          boundaryGap: false,
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          textStyle: {
            fontSize: 15,
            color: "#000000FF"
          },
          data: timeline,
          gridIndex: 5}
        ],
        yAxis: [
          {type: 'value',
          name: false,
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          axisLabel: {
            margin: 10,
            textStyle: {
              fontSize: 14
            }
          },
          splitLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          gridIndex: 0},
          {type: 'value',
          name: false,
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          axisLabel: {
            margin: 10,
            textStyle: {
              fontSize: 14
            }
          },
          splitLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          gridIndex: 1},
          {type: 'value',
          name: false,
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          axisLabel: {
            margin: 10,
            textStyle: {
              fontSize: 14
            }
          },
          splitLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          gridIndex: 2},
          {type: 'value',
          name: false,
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          axisLabel: {
            margin: 10,
            textStyle: {
              fontSize: 14
            }
          },
          splitLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          gridIndex: 3},
          {type: 'value',
          name: false,
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          axisLabel: {
            margin: 10,
            textStyle: {
              fontSize: 14
            }
          },
          splitLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          gridIndex: 4},
          {type: 'value',
          name: false,
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          axisLabel: {
            margin: 10,
            textStyle: {
              fontSize: 14
            }
          },
          splitLine: {
            lineStyle: {
              color: '#ADADADFF'
            }
          },
          gridIndex: 5}
        ],
        series: [{
            name: "AQI",
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            },
            itemStyle: {
              normal: {
                color: 'rgb(255,48,48)',
                borderColor: 'rgba(0,136,212,0.2)',
                borderWidth: 12
              }
            },
            data: a1,
            xAxisIndex: 0,
            yAxisIndex: 0
          },
          {
            name: "no2",
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            },
            itemStyle: {
              normal: {
                color: 'rgb(139,126,102)',
              }
            },
            data: a2,
            xAxisIndex: 1,
            yAxisIndex: 1
          },
          {
            name: "o3",
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            },
            itemStyle: {
              normal: {
                color: 'rgb(34,139,34)',
              }
            },
            data: a3,
            xAxisIndex: 2,
            yAxisIndex: 2
          },
          {
            name: "pm10",
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            },
            itemStyle: {
              normal: {
                color: 'rgb(255 255 0)',
              }
            },
            data: a4,
            xAxisIndex: 3,
            yAxisIndex: 3
          },
          {
            name: "pm25",
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            },
            itemStyle: {
              normal: {
                color: 'rgb(106,90,205)',
              }
            },
            data: a5,
            xAxisIndex: 4,
            yAxisIndex: 4
          },
          {
            name: "so2",
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            },
            itemStyle: {
              normal: {
                color: 'rgb(0,0,205)',
              }
            },
            data: a6,
            xAxisIndex: 5,
            yAxisIndex: 5
          }
        ]
      };
      this.Chart.setOption(option);
    }
  },
}

</script>
<style>
.map-class {
  width: 100%;
  height: 100%;
}

.anchorBL {
  display: none;
}

</style>
