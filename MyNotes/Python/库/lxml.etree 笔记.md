# lxml.etree笔记

概览:

lxml.etree中提供了几种方法用于解析文本:

- fromstring() : 用于解析字符串
- HTML()： 用于解析HMTL对象
- XML(): 用于解析XML对象
- parse(): 用于解析文件类型的对象

### 1. HMTL()

```python
text = '''
<div>
    <ul>
         <li class="item-0"><a href="link1.html">first item</a></li>
         <li class="item-1"><a href="link2.html">second item</a></li>
         <li class="item-inactive"><a href="link3.html">third item</a></li>
         <li class="item-1"><a href="link4.html">fourth item</a></li>
         <li class="item-0"><a href="link5.html">fifth item</a>
     </ul>
 </div>
'''
html = etree.HTML(text)
result = etree.tostring(html)
print(result)
```

输出结果:

```xml
<html><body>
<div>
    <ul>
         <li class="item-0"><a href="link1.html">first item</a></li>
         <li class="item-1"><a href="link2.html">second item</a></li>
         <li class="item-inactive"><a href="link3.html">third item</a></li>
         <li class="item-1"><a href="link4.html">fourth item</a></li>
         <li class="item-0"><a href="link5.html">fifth item</a></li>
</ul>
 </div>
 
</body></html>
```

可以看到，lxml.HTML()具有自动修正HTML代码的功能，不仅补全了li标签(最后一个)，还自动添加了html, body标签。

### 2. parse()解析文件

把上面的文件存为test.html:

```python
from lxml import etree
html = etree.parse('test.html')
result = etree.tosring(html, pretty_print=True)
print result
```

与用HTML()得到的结果一样。

### 3. XPath实例测试

获取所有的`<li>`标签

```python
from lxml import etree
html = etree.parse('test.html')
print type(html)  # <type 'lxml.etree._ElementTree'>

result = html.xpath('//li')
```

上面的html是一个`ElementTree`对象，用xpath解析后，得到的结果都是list。list中的元素为`Element`对象。

因此，`Element`对象都可以用`.tag`查看tag，用`.attrib`查看属性，用`.text`查看text元素。

其他例子：

- 获取所有`<li>`标签下的class:

```python
result = html.xpath('//li/@class')
```

返回结果:

```
['item-0', 'item-1', 'item-inactive', 'item-1', 'item-0']
```

注意什么时候返回包含str的list，什么时候返回包含Element元素的list

- 获取`<li>`标签下href为link1.html的`<a>`标签:

```python
result = html.xpath('//li/a[@href="link1.html"]')
```

- 获取`<li>`标签下的所有`<span>`标签

```python
result = html.xpath('//li//span')
```

注意，因为span并不是li的子元素，所有不能用`//li/span`

- 获取class为bold的标签名

```python
result = html.xpath('//*[@class="bold"]')
```

- 用text()获取当前节点下的文本元素

```python
result = html.xpath('//*[@class]/text()')
# ['third item']
result = html.xpath('//*[@class]/*/text()')
# ['first item', 'second item', 'fourth item', 'fifth item']
result = html.xpath('//*[@class]/text() | //*[@class]/*/text()')
result = html.xpath('//li//text()')
# 上面两个结果一样，都是
# ['first item', 'second item', 'third item', 'fourth item', 'fifth item']
```

### 4. 技巧性实例

对这样一段html：

```html
<ul id="parameter2" class="p-parameter-list">
 <li title='养生堂天然维生素E软胶囊'>商品名称：养生堂天然维生素E软胶囊</li>
 <li title='720135'>商品编号：720135</li>
 <li title='养生堂'>品牌：<a>养生堂</a></li>
</ul>
```

我们这么要提取li标签的**完整**文字，且list长度为3，怎么处理？

方法是：用string()获得nested节点文字内容：

```python
# coding: utf-8
from lxml import etree
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
t = '''
<ul id="parameter2" class="p-parameter-list">
 <li title='养生堂天然维生素E软胶囊'>商品名称：养生堂天然维生素E软胶囊</li>
 <li title='720135'>商品编号：720135</li>
 <li title='养生堂'>品牌：<a>养生堂</a></li>
</ul>
'''
html = etree.HTML(t.decode('utf-8'))  # 注意，有中文，因此要先decode
l = html.xpath('//ul[@id="parameter2"]/li')
for i in l:
    print i.xpath('string(.)')  # 当前节点下所有深层的text都提取出来
print len(l)
```

输出结果: 

```
商品名称：养生堂天然维生素E软胶囊
商品编号：720135
品牌：养生堂
3
```

或者用另外一种方法手动实现：

```python
l = html.xpath('//ul[@id="parameter2"]/li')

def tryFindChild(element):
    children = element.getchildren()
    if children:
        return element.text + ' ' + children[0].text
    return element.text

for i in l:
    print tryFindChild(i)

print len(l)
```

-----

lxml中还有一些很有用的方法如remove等等







