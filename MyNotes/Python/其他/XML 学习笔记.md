# XML学习笔记

### 1. 基本概念

一个简单的XML例子:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<bookstore>
    <book category="COOKING">
        <title lang="en">Everyday Italian</title>
        <author>Giada De Laurentiis</author>
        <year>2005</year>
        <price>30.00</price>
    </book>
    <book category="CHILDREN">
        <title lang="en">Harry Potter</title>
        <author>J K. Rowling</author>
        <year>2005</year>
        <price>29.99</price>
    </book>
    <book category="WEB">
        <title lang="en">Learning XML</title>
        <author>Erik T. Ray</author>
        <year>2003</year>
        <price>39.95</price>
    </book>
</bookstore>
```

第一行为XML声明: XML版本为1.0， 使用的编码为utf-8。声明为可选部分。

第二行为描述文档的**根元素**`<bookstore>`，一个XML文档必须包含根元素。

对于`book`: 

`category`叫**属性**，由key=value形式组成，value**必须加引号**；而前后两个标签之间的内容如Everyday Italian，叫**元素**。

XML中的注释方式: `<!-- this is a comment -->`

在XML中，文档中的空格会被保留，这点与HTML不同。

XML以LF存储换行(Win: CRLF, Unix 和 macOS: LF, 旧的OSX: CR)。


### 2. 实体引用

在XML中，一些具有特殊意义的字符必须用实体引用替代。比如，如果把字符"<"放在XML元素中会发生错误，因为解析器会把它当做新元素的开始。

XML中的实体引用：

![](https://ww3.sinaimg.cn/large/006tNc79ly1fliq6p62rjj30ky0c8jsb.jpg)
在 XML 中，只有字符 "<" 和 "&" 确实是非法的。大于号是合法的，但是用实体引用来代替它是一个好习惯。


### 3. XPath语法

```xml
<?xml version="1.0" encoding="UTF-8"?>

<bookstore>

<book>
  <title lang="eng">Harry Potter</title>
  <price>29.99</price>
</book>

<book>
  <title lang="eng">Learning XML</title>
  <price>39.95</price>
</book>

</bookstore>
```

#### 3.1 选取节点

| 表达式      | 描述                |
| -------- | ----------------- |
| nodename | 选取此节点的所有子节点。      |
| /        | 从根节点选取            |
| //       | 从匹配的当前节点中选取，不考虑位置 |
| .        | 选取当前节点            |
| ..       | 选取当前节点的父节点        |
| @        | 选取属性              |

举例:

| 路径表达式           | 结果                                       |
| --------------- | ---------------------------------------- |
| bookstore       | 选取bookstore元素的所有子节点                      |
| /bookstore      | 选取根元素bookstore。路径起始于/表示从走绝对路径            |
| bookstore/book  | 选取属于bookstore的子元素的所有book元素               |
| //book          | 选取所有的book子元素，而不考虑它们在文档中的位置               |
| bookstore//book | 选取bookstore后代的所有book元素，不考虑它们在bookstore之下的什么位置 |
| //@lang         | 选取名为lang的所有属性                            |

#### 3.2 谓语(Predicated)

嵌套在方括号中，用来查找某个特定的节点。

| 路径表达式                              | 结果                                       |
| ---------------------------------- | ---------------------------------------- |
| /bookstore/book[1]                 | 选取属于 bookstore 子元素的第一个 book 元素。          |
| /bookstore/book[last()]            | 选取属于 bookstore 子元素的最后一个 book 元素。         |
| /bookstore/book[last()-1]          | 选取属于 bookstore 子元素的倒数第二个 book 元素。        |
| /bookstore/book[position()<3]      | 选取最前面的两个属于 bookstore 元素的子元素的 book 元素。    |
| //title[@lang]                     | 选取所有拥有名为 lang 的属性的 title 元素。             |
| //title[@lang='eng']               | 选取所有 title 元素，且这些元素拥有值为 eng 的 lang 属性。   |
| /bookstore/book[price>35.00]       | 选取 bookstore 元素的所有 book 元素，且其中的 price 元素的值须大于 35.00。 |
| /bookstore/book[price>35.00]/title | 选取 bookstore 元素中的 book 元素的所有 title 元素，且其中的 price 元素的值须大于 35.00。 |

#### 3.3 通配符

| 通配符    | 描述        |
| ------ | --------- |
| *      | 匹配任何元素节点  |
| @*     | 匹配任何属性节点  |
| node() | 匹配任何类型的节点 |

举例:

| 路径表达式        | 结果                  |
| ------------ | ------------------- |
| /bookstore/* | 选取bookstore元素的所有子元素 |
| //*          | 选取文档中的所有元素          |
| //title[@*]  | 选取所有带有属性的title元素    |

#### 3.4 选取若干路径

可用`|`选取多个路径：

比如`//book/title | //book/price`表示选取 book 元素的所有 title 和 price 元素。<!---->

=====
http://www.runoob.com/xml/xml-tutorial.html

