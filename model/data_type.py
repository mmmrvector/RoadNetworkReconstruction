import math
#先定一个node的类
class Node():                  #value + next
    def __init__ (self, value = None, next = None):
        self._value = value
        self._next = next

    def getValue(self):
        return self._value

    def getNext(self):
        return self._next

    def setValue(self,new_value):
        self._value = new_value

    def setNext(self,new_next):
        self._next = new_next

#实现Linked List及其各类操作方法
class LinkedList():
    def __init__(self):      #初始化链表为空表
        self._head = Node()
        self._tail = None
        self._length = 0



    #检测是否为空
    def isEmpty(self):
        return self._head is None

    #add在链表前端添加元素:O(1)
    def add(self,value):
        newnode = Node(value,None)    #create一个node（为了插进一个链表）
        newnode.setNext(self._head)
        self._head = newnode
        self._length += 1

    #append在链表尾部添加元素:O(n)
    def append(self,value):
        self._length += 1
        newnode = Node(value)
        if self.isEmpty():
            self._head = newnode   #若为空表，将添加的元素设为第一个元素
        else:
            current = self._head
            while current.getNext() is not None:
                current = current.getNext()   #遍历链表
            current.setNext(newnode)   #此时current为链表最后的元素

    #union  合并两个链表
    def union(self, link2):
        head = link2._head.getNext()
        tail = self._head.getNext()
        while tail.getNext() is not None:
            tail = tail.getNext()
        tail.setNext(head)
        self._length += link2._length

    #max_path_segment   this is for debug
    def max_path_segment(self):
        node1 = self._head.getNext()
        node2 = node1.getNext()
        max_dis = 0
        while node2 is not None:
            x1 = node1.getValue()
            x2 = node2.getValue()
            dis = math.sqrt(math.pow(x1[0] - x2[0],2) + math.pow(x1[1] - x2[1],2))
            max_dis = (dis if dis > max_dis else max_dis)
            node1 = node2
            node2 = node1.getNext()
        return max_dis


    #search检索元素是否在链表中
    def search(self,value):
        current=self._head
        foundvalue = False
        while current is not None and not foundvalue:
            if current.getValue() == value:
                foundvalue = True
            else:
                current=current.getNext()
        return foundvalue

    #index索引元素在链表中的位置
    def index(self,value):
        current = self._head
        count = -1
        found = None
        while current is  not None and not found:
            count += 1

            if current.getValue()==value:
                found = True
            else:
                current=current.getNext()
        if found:
            return count
        else:
            raise ValueError ('%s is not in linkedlist'%value)

    #remove删除链表中的某项元素
    def remove(self,value):
        current = self._head
        pre = None
        while current is not None:
            if current.getValue() == value:
                if not pre:
                    self._head = current.getNext()
                else:
                    pre.setNext(current.getNext())
                break
            else:
                pre = current
                current = current.getNext()
    #deepcopy使用deepcopy前，self必须为空链表
    def deep_copy(self, link_list):
        temp_node = link_list._head.getNext()
        while temp_node is not None:
            temp_point = temp_node.getValue()
            self.append(temp_point)
            temp_node = temp_node.getNext()
        self._tail = None

    #get_same_segment 返回重合路段
    def get_same_segment(self, link_list):
        ans_list = []
        self_node1 = self._head.getNext()

        while self_node1.getNext() is not None:
            self_node2 = self_node1.getNext()
            link_node1 = link_list._head.getNext()
            while link_node1.getNext() is not  None:
                link_node2 = link_node1.getNext()
                if self_node1.getValue() == link_node1.getValue() and self_node2.getValue() == link_node2.getValue():
                        ans_list.append(self_node1.getValue())
                        break
                        #ans_list.append(self_node2.getValue())
                link_node1 = link_node1.getNext()
            self_node1 = self_node1.getNext()
        return ans_list

    # split 将路段以某一基础路段为分界，分裂为两个路段
    def split(self, point):
        left_part = LinkedList()
        right_part = LinkedList()
        node = self._head.getNext()
        flag = False
        while node is not None:
            if node.getValue() == point:
                left_part.append(node.getValue())
                if node.getNext() is not None:
                    # right_part.append(node.getNext().getValue())
                    flag = True
                    node = node.getNext()
            else:
                if flag:
                    right_part.append(node.getValue())
                else:
                    left_part.append(node.getValue())
                node = node.getNext()
        if left_part._length == 0 or left_part._length == 1:
            left_part = None
        if right_part._length == 0 or right_part._length == 1:
            right_part = None
        return left_part, right_part

    # insert链表中插入元素
    def insert(self,pos,value):
        if pos <= 1:
            self.add(value)
        elif pos > self.size():
            self.append(value)
        else:
            temp = Node(value)
            count = 1
            pre = None
            current = self._head
            while count < pos:
                count += 1
                pre = current
                current = current.getNext()
            pre.setNext(temp)
            temp.setNext(current)

    def to_list(self):
        ans = []
        node = self._head.getNext()
        while node is not None:
            point = node.getValue()
            ans.append(point)
            node = node.getNext()
        return ans

    def __str__(self):
        if self._length == 0:
            return "LinkedList is None"
        str1 = ""
        node = self._head.getNext()
        while node != None:
            str1 =str1 +" "+ str(node.getValue())
            node = node.getNext()
        return str1

    def __eq__(self, other):

        if other is None:
            if self is None:
                return True
            else:
                return False

        flag = True
        node1 = self._head.getNext()
        node2 = other._head.getNext()
        while node1!= None and node2 != None:
            if node1.getValue() !=  node2.getValue():
                flag = False
                break
            node1 = node1.getNext()
            node2 = node2.getNext()
        if node1 != None or node2 != None:
            flag = False
        return flag