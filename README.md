# Conditional-Random-Field
<ol>
  <li>简介</li>使用Conditional Random Field模型预测当前句子的词性序列
  <li>Conditional Random Field模型</li>
  <ul>Log Linear Model与HMM模型的结合
    <li>特征提取</li>添加一个新特征：当前词性+前一个词性
    <li>权重学习</li>同Log Linear Model的在线学习方法
    <li>目标函数</li>最大熵模型（一个句子为一个整体）
    <li>句子最佳词性序列</li>同HMM模型（维特比算法）
  </ul>
  <li>评价指标</li>准确率 = 正确的标注数 / 总的标注数
  <li>程序</li>
  <ol>
    <li>数据</li>
    训练集：train.conll<br>
    测试集：dev.conll
    <li>代码</li>
    conditional_random_field.py:实现Conditional Random Field模型词性标注
  </ol>
</ol>

