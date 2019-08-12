//
// Created by xiping on 2019-06-04.
//

#ifndef LEETCODE_QUESTION_H
#define LEETCODE_QUESTION_H

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <algorithm>
using namespace std;

void rotate(vector<int>& nums, int k) {
    int len = nums.size();
    if(len ==0)
        return;
    k = k % len;
    int temp,s,j;
    for (int i = k; i > 0; i--) {
        s = len-i;
        temp = nums[s];
        for (j = s; j > k-i; j--) {
            nums[j] = nums[j-1];
        }
        nums[j] = temp;
    }
}

//这种方法不可行，当2*k>len时，会出现越界的错误
void rotate_improve(vector<int>& nums, int k) {
    int len = nums.size();
    if(len ==0)
        return;
    k = k % len;
    int tem;
    for (int l = 0; l < k; ++l) {
        tem = nums[l];
        nums[l] = nums[len-k+l];
        nums[len-k+l] = tem;
    }

    int temp,s,j;
    for (int i = 0; i < k; i++) {
        s = i+k+1;
        temp = nums[s];
        for (j = s; j > k+i; j--) {
            nums[j] = nums[j-1];
        }
        nums[j] = temp;
    }
}

//使用翻转的方法
//时间复杂度为o(n)
//空间复杂度为o(1)
void rotate_reversal(vector<int>& num, int k){
    //需要进行多次

    int len = num.size();
    k = k % len;
    int temp;
    for (int i = 0; i < len/2; ++i) {
        temp = num[i];
        num[i] = num[len-i-1];
        num[len-i-1] = temp;
    }
    for (int j = 0; j < k/2; ++j) {
        temp = num[j];
        num[j] = num[k-j-1];
        num[k-j-1] = temp;
    }
    for (int l = k; l < k+(len-k)/2; ++l) {
        temp = num[l];
        num[l] = num[len+k-l-1];
        num[len+k-l-1] = temp;

    }
}

bool containsDuplicate(vector<int>& nums) {

    int max=nums[0];
    for (int i = 1; i < nums.size(); ++i) {
        if(nums[i]>max)
            max = nums[i];
    }
    int* flag = new int[max+1];
    for (int k = 0; k <= max; ++k) {
        flag[k] = 0;
    }

    for (int j = 0; j < nums.size(); ++j) {
        if(flag[nums[j]]==1){
            return true;
        }else{
            flag[nums[j]]=1;
        }
    }
    return false;
}


void shiftDown(vector<int>& nums, int k, int len){
    while(2*k+1<len){
        int j = 2*k+1;
        if(j+1<len && nums[j]<nums[j+1])
            j++;
        if(nums[k]<nums[j])
            swap(nums[k],nums[j]);
        k = j;
    }
}

bool containsDuplicate_i(vector<int>& nums) {
    //先进行排序，然后再遍历一遍数组


    //原地堆排序
    //建堆
    int len = nums.size();
    for (int i = (len-2)/2; i >= 0; i--) {
        shiftDown(nums,i,len);
    }
    //排序
    for (int j = 0; j < len; ++j) {
        swap(nums[0],nums[len-j-1]);
        shiftDown(nums,0,len-j-1);
    }
    //找相同元素
    int temp = nums[0];
    for (int k = 1; k < len; ++k) {
        if(nums[k]!=temp)
            temp = nums[k];
        else
            return true;
    }
    return false;
}

//只存在一次的数字
int singleNumber(vector<int>& nums){
    int a = 0;
    for (int i = 0; i < nums.size(); ++i) {
        a = a ^ nums[i];
    }
    return a;
}

//两个数组的交集
vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
    //使用Map数据结构
    map<int,int> df;
    vector<int> result;

    for (int i = 0; i < nums1.size(); ++i) {
        if(df[nums1[i]]==0)
            df[nums1[i]] = 1;
        else
            df[nums1[i]]++;
    }

    for (int j = 0; j < nums2.size(); ++j) {
        if(df[nums2[j]] > 0){
            result.push_back(nums2[j]);
            df[nums2[j]]--;
        }
    }
    return result;
}

//加1
vector<int> plusOne(vector<int>& digits) {

    int carry = 1;
    int i;
    for (i = digits.size()-1; i >= 0; i=i-1) {
        digits[i] = digits[i] + carry;
        if(digits[i] == 10) {
            digits[i] = digits[i] - 10;
            carry = 1;
        } else {
            carry = 0;
            break;
        }
    }
    if(i==-1 && carry == 1){ //最高位有进位
        digits[0] = 1;
        for (int j = 1; j < digits.size(); ++j) {
            digits[j] = 0;
        }
        digits.push_back(0);
    }
    return digits;
}

//将所有0移动到数组的末尾，同时保持非零元素的相对顺序
void moveZeroes(vector<int>& nums) {
    int j = 0;
    int count = 0;
    int len = nums.size();
    for (int i = 0; i < len; ++i) {
        if(nums[i]!=0){
            nums[j] = nums[i];
            j++;
        }else{
            count++;
        }
    }
    for (int k = len-1; k >= len-count; --k) {
        nums[k] = 0;
    }
}

//对nums[l...r]进行划分
int partition(vector<int>& nums, int l, int r, vector<int>& index){
    int ra = l + rand()%(r-l+1);
    swap(nums[l],nums[ra]);
    swap(index[l],index[ra]);
    int pivot = nums[l];
    int i=l+1,j=r;
    while(i<j){
     while(nums[j]>=pivot && i<j) j--;
     while(nums[i]<=pivot && i<j) i++;
     swap(nums[i],nums[j]);
     swap(index[i],index[j]);
    }
    if(nums[l] > nums[i]) {
        swap(nums[l], nums[i]);
        swap(index[l],index[i]);
    }
    return i;

}

void quickSort(vector<int>& nums, int l, int r, vector<int>& index){
    if(l<r){
        int partit = partition(nums,l,r,index);
        quickSort(nums,l,partit-1,index);
        quickSort(nums,partit+1,r,index);
    }
}

//二分搜索算法，递归
int binartSearch(vector<int>& nums, int l, int r, int target){
    if (l>r)
        return -1;
    int mid = l + (r-l)/2;
    if(nums[mid]==target)
        return mid;
    else if(nums[mid]<target)
        return binartSearch(nums,mid+1,r,target);
    else
        return binartSearch(nums,l,mid-1,target);
}

vector<int> twoSum(vector<int>& nums, int target) {
    //先排好序，再使用两个循环找出相应的数组下标
    //快排
    vector<int> res;
    srand(time(NULL));
    int j=nums.size()-1;
    int temp,loc;
    vector<int> index;
    for (int i = 0; i < j+1; ++i) {
        index.push_back(i);
    }
    quickSort(nums,0,j,index);
    for (int k = 0; k<j; ++k) {
        temp = target - nums[k];
        //二分搜索nums[k+1,j];
        loc = binartSearch(nums,k+1,j,temp);
        if(loc!=-1){
            res.push_back(index[k]);
            res.push_back(index[loc]);
            break;
        }
    }
    return res;
}

vector<int> two_sums(vector<int>& nums, int target){
    unordered_map<int,int> hash;
    vector<int> res;
    for (int i = 0; i < nums.size(); ++i) {
        if(hash.find(target-nums[i])==hash.end()){
            hash[nums[i]] = i;
            continue;
        }
        res.push_back(i);
        res.push_back(hash[target-nums[i]]);
        return res;
    }
    return res;
}

void initialToZero(map<char,int> ele){
    for (int i = 0; i < ele.size(); ++i) {
        ele[i+'1'] = 0;
    }
}

//有效的数独
bool isValidSudoku(vector<vector<char>>& board) {
    map<char,int> elements;
    initialToZero(elements);
    //检查数独中的每一行是否满足条件
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[0].size(); ++j) {
            if(board[i][j]=='.'){
                continue;
            }else if(elements[board[i][j]]==1){
                return false;
            }else{
                elements[board[i][j]]++;
            }
        }
        initialToZero(elements);
    }
    //检查数独中的每一列是否满足条件
    for (int k = 0; k < board.size(); ++k) {
        for (int i = 0; i < board[0].size(); ++i) {
            if(board[i][k]=='.'){
                continue;
            }else if(elements[board[i][k]]==1){
                return false;
            }else{
                elements[board[i][k]]++;
            }
        }
        initialToZero(elements);
    }

    //3*3宫内是否满足条件
    for (int l = 0; l <= 6; l=l+3) { //行向九宫格
        for (int i = 0; i <=6 ; i=i+3) {    //列向九宫格
            //九宫格内循环
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    if(board[l+j][i+k]=='.'){
                        continue;
                    }else if(elements[board[l+j][i+k]]==1)
                        return false;
                    else
                        elements[board[l+j][i+k]]++;
                }
            }
            initialToZero(elements);
        }
    }

    return true;

}


// Definition for a binary tree node.
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};


//给定一个二叉树，找出其最大深度
//递归的方式求二叉树
int maxDepth(TreeNode* root) {
    if(root==NULL)
        return 0;
    int left_maxDepth = maxDepth(root->left);
    int right_maxDepth = maxDepth(root->right);
    return left_maxDepth>right_maxDepth? (left_maxDepth+1):(right_maxDepth+1);
}


//验证一颗树是否是二叉搜索树
//可以通过中序遍历方式遍历一遍二叉搜索树，若有序，则是一颗二叉搜索树
//直接递归的方式不行，通过遍历的方式

void inOrder(TreeNode* root,vector<int>& q){
    if(root!=NULL){
        if(root->left)
            inOrder(root->left,q);
        q.push_back(root->val);
        if(root->right){
            inOrder(root->right,q);
        }
    }
}

bool isValidBST(TreeNode* root) {
    vector<int> q;
    inOrder(root,q);
    if(q.empty())
        return true;
    for (int i = 0; i < q.size()-1; ++i) {
        if(q[i+1]>q[i])
            continue;
        else
            return false;
    }
    return true;
}


//最大子序和
int maxSubArray_old(vector<int>& nums) {
    vector<int> a1,a2,a3;
    //辅助数组的初始化
    for (int i = 0; i < nums.size(); ++i) {
        a1.push_back(nums[i]);
        a2.push_back(i);

    }
    //遍历计算以a1数组中每个元素作为起点的最大子序和
    for (int j = 1; j < nums.size(); ++j) {
        for (int i = 0; i < j; ++i) {
            nums[i] = nums[i] + nums[j];
            if(nums[i]>a1[i]){
                a1[i] = nums[i];
                a2[i] = j;
            }
        }
    }
    //查找a1,找到最大的值
    int max=a1[0];
    for (int k = 1; k < a1.size(); ++k) {
        if(a1[k]>max)
            max = a1[k];
    }
    return max;
}

//最大子序和，基于动态规划的做法
//f(n),以第n个数为结束点的子数列的最大和，则存在一个递推关系 f(n)=max(f(n-1)+a[n],a[n])
int maxSubArray(vector<int>& nums) {
    int len = nums.size();
    if(len == 0)
        return 0;
    int fn = nums[0];
    int max = fn;
    for (int i = 1; i < len; ++i) {
        if(fn+nums[i]>nums[i]) {
            fn = fn + nums[i];
        } else
            fn = nums[i];
        if(fn>max)
            max = fn;
    }
    return max;
}

//爬楼梯问题
int climbStairs(int n) {
    if(n==0)
        return 0;
    if(n==1)
        return 1;
    vector<int> a;
    a.push_back(1); //a[0] = 1;
    a.push_back(1); //a[1] = 1;
    for (int i = 2; i <= n; ++i) {
        a.push_back(a[i-1]+a[i-2]);
    }
    return a[n];
}


//单链表相关问题

//Definition for singly-linked list.
 struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
 };


bool isPalindrome(ListNode* head) {
    //遍历一次获得列表的长度
    ListNode *p = head;
    int len = 0;
    while(p!=NULL){
        len++;
        p = p->next;
    }
    int mid = len/2;
    //使用栈保存遍历的元素
    p = head;
    stack<int> s;
    for (int i = 0; i < mid; ++i) {
        s.push(p->val);
        p=p->next;
    }
    if(len%2==1)
        p=p->next;
    //进行比较
    while(p!=NULL){
        if(p->val == s.top()){
            s.pop();
            p = p->next;
            continue;
        }
        else
            return false;
    }
    return true;
}
//使用O(n)的时间复杂度和O(1)的空间复杂度完成回文链表的判断
//1.使用快慢指针找到链表中点；
//2.逆转链表后半部分；
//3.从头和中点，开始比较是否相同；
bool isPalindrome_improve(ListNode* head) {
    //1.快慢指针找到中间节点,slow指针最终位置就是链表中间节点的位置
    if(head==NULL)
        return false;
    ListNode* fast = head;
    ListNode* slow = head;
    while(fast){
        slow = slow->next;
        fast = fast->next ? fast->next->next : fast->next;
    }
    //slow 所指的节点为中间节点
    //对后半部分指针进行逆序
    //post所指为链表最后一个元素
    ListNode* pre = nullptr;
    ListNode* post = slow;
    ListNode* temp = post->next;
    while(temp){
        post->next = pre;
        pre = post;
        post = temp;
        temp = temp->next;
    }
    post->next = pre;
    //前后比较判断是否是回文链表
    while(post){
        if(post->val != head->val)
            return false;
        else{
            post = post->next;
            head = head->next;
        }
    }
    return true;
}

//一趟扫描完成删除链表的倒数第N个节点的操作
ListNode* removeNthFromEnd(ListNode* head, int n) {
    //使用快慢指针来计算整条链表的长度，再通过n值，从head或从slow开始找到最终要删除的节点
    ListNode *slow = head, *fast = head;
    int flag = 0; //标记是偶数还是奇数
    int count = 0;  //记录slow走过的长度
    while(fast){
        slow = slow->next;
        count++;
        if(fast->next){
            fast = fast->next->next;
        }
        else{
            fast = fast->next;
            flag = 1;
        }
    }
    int len = count*2-flag;

    int step;
    if(n<len/2){
        step = len/2-n-1;
        while(step){
            slow = slow->next;
            step--;
        }
        ListNode *temp = slow->next;
        slow->next = slow->next->next;
        delete(temp);
    }else if(n == len){
        head = head->next;
        return head;
    }else{
        fast = head;
        step = len - n - 1;
        while(step){
            fast = fast->next;
            step--;
        }
        ListNode * temp = fast->next;
        fast->next = fast->next->next;
        delete(temp);
    }
    return head;
}

//Tengxun
//1.爬塔问题
int time_fly(vector<int>& a){
    int len = a.size();
    if(len==0)
        return 0;
    vector<int> f,y;
    f.push_back(0);//f(0)=0;
    f.push_back(0);//f(1)=0;
    y.push_back(0);//y(0)=0;
    y.push_back(a[0]);//y(1)=0;
    for (int i = 2; i <= len; ++i) {
        y.push_back(f[i-1]+a[i-1]);
        f.push_back(y[i-1]<y[i-2]?y[i-1]:y[i-2]);
    }
    return f[len]<y[len]?f[len]:y[len];
}

//合并两个有序数组
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    //使用一个辅助数组，然后一一比较
    vector<int> nums3;
    for (int i = 0; i < m; ++i) {
        nums3.push_back(nums1[i]);
    }
    int i=0,j=0,k=0;
    while(i<m && j<n){
        if(nums3[i]<nums2[j]){
            nums1[k] = nums3[i];
            i++;
        }else{
            nums1[k] = nums2[j];
            j++;
        }
        k++;
    }
    while(i<m){
        nums1[k] = nums3[i];
        i++;
        k++;
    }
    while(j<n){
        nums1[k] = nums2[j];
        j++;
        k++;
    }

}

//数组中的第K个最大元素
//使用堆排序来做,构建一个包含k个元素的最小堆，然后更新堆中的元素，等遍历完nums中的元素，堆顶就是所要求得第k个元素
int findKthLargest(vector<int>& nums, int k) {
    vector<int> minHead;
    for (int i = 0; i < k; ++i) {
        minHead.push_back(nums[i]);
    }
    //建堆
    make_heap(minHead.begin(),minHead.end(),greater<int>());

    //遍历nums中剩余元素
    minHead.push_back(0);
    for (int l = k; l < nums.size(); ++l) {
        if(nums[l]>minHead[0]){
            minHead[k] = nums[l];
            pop_heap(minHead.begin(),minHead.end(),greater<int>());
        }
    }
    return minHead[0];
}


//找两个单链表相交的起始节点
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    //先计算两条链表的长度差c，较长的链表先走c步，然后两条链表一起遍历，若遇到地址相同的节点则是公共的节点
    if(headA==NULL || headB==NULL)
        return NULL;
    ListNode *headA_temp = headA, *headB_temp = headB;
    int lenA=0,lenB=0;
    while(headA_temp){
        lenA++;
        headA_temp = headA_temp->next;
    }
    while(headB_temp){
        lenB++;
        headB_temp = headB_temp->next;
    }
    int len_diff;
    if(lenA>lenB){
        len_diff = lenA - lenB;
        while(len_diff){
            headA = headA->next;
            len_diff--;
        }
    }else{
        len_diff = lenB - lenA;
        while(len_diff){
            headB = headB->next;
            len_diff--;
        }
    }
    while(headA){
        if(headA == headB)
            return headA;
        else{
            headA = headA->next;
            headB = headB->next;
        }
    }
    return NULL;
}

//给定一个链表，返回链表开始入环的第一个节点
ListNode *detectCycle(ListNode *head) {
    if(head==NULL)
        return NULL;
    //快慢指针判断是否有环
    ListNode *fast = head, *slow = head;
    while(fast){
        slow = slow->next;
        if(fast->next == NULL || fast->next->next == NULL){
            return NULL;
        }else{
            fast = fast->next->next;
            if(fast == slow)
                break;
        }
    }
    //相交的第一个节点即为开始入环的第一个节点
    fast = head;
    while(fast){
        if(slow==fast){
            break;
        }
        else{
            fast = fast->next;
            slow = slow->next;
        }
    }
    return slow;
}

//使用优先队列建堆

int findKthLargest_priory_queue(vector<int>& nums, int k) {
    //相当于是构建最小堆
    if(nums.empty())
        return 0;
    priority_queue<int,vector<int>,greater<int>> heap;
    for (int i = 0; i < k; ++i) {
        heap.push(nums[i]);
    }
    for (int j = k; j < nums.size(); ++j) {
        if(heap.top()<nums[j]){
            heap.pop();
            heap.push(nums[j]);
        }
    }
    return heap.top();
}

//递归求全排列
//从哪一个值开始，flag矩阵，以遍历个数
void permute_loop(int i, vector<int>& flag, int count, vector<int>& temp, int total_length,vector<int>& nums){
    flag[i] = 0;
    count++;
    if(count < total_length){
        for (int j = 0; j < total_length; ++j) {
            if(flag[j]==0){
                temp.push_back(nums[j]);
                permute_loop(j, flag, count, temp, total_length, nums);
            }
        }
    }
}


//回溯算法求一个没有重复数字的序列的全排列
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> temp, flag(nums.size(),0);//flag标志某一个元素是否走过
    int count = 0;
    if(nums.empty())
        return result;
    for (int i = 0; i < nums.size(); ++i) {
        temp.push_back(nums[i]);
        permute_loop(i, flag, count, temp, nums.size(),nums);
    }
    return result;
}

TreeNode* reConstructBinaryTree_re(vector<int> pre, vector<int> vin, int pre_l, int pre_r \
,int vin_l, int vin_r) {
    //TreeNode node(0);
    TreeNode* root = new TreeNode(0);
    root->val = pre[pre_l];
    if(pre_l==pre_r){
        return root;
    }
    //在中序遍历序列中找到根节点的位置
    int loc = -1;
    for (int i = vin_l; i <= vin_r; ++i) {
        if(vin[i] == pre[pre_l]){
            loc = i;
            break;
        }
    }
    //构建左子树
    if(loc>vin_l)
        root->left = reConstructBinaryTree_re(pre,vin,pre_l+1,pre_l+loc-vin_l,vin_l,loc-1);
    else
        root->left = nullptr;
    //构建右子树
    if(loc<vin_r)
        root->right = reConstructBinaryTree_re(pre,vin,pre_l+loc-vin_l+1,pre_r,loc+1,vin_r);
    else
        root->right = nullptr;

    return root;

}

//重建二叉树
TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
    int len = pre.size();
    return reConstructBinaryTree_re(pre, vin, 0, len-1, 0, len-1);
}


//用两个栈实现一个队列
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if(stack2.empty()){
            //将stack1中的内容移到stack2中
            if(stack1.empty())
                throw "The queue is empty.";
            else {
                while(!stack1.empty()){
                    stack2.push(stack1.top());
                    stack1.pop();
                }
            }
        }
        int res = stack2.top();
        stack2.pop();
        return res;

    }

private:
    stack<int> stack1;  //用于入栈
    stack<int> stack2;  //用于出栈
};


//二叉树的镜像
void Mirror(TreeNode *pRoot) {
    if(pRoot== nullptr)
        return;
    TreeNode *temp = pRoot->left;
    pRoot->left = pRoot->right;
    pRoot->right = temp;
    Mirror(pRoot->left);
    Mirror(pRoot->right);
}

//顺时钟打印矩阵
//四个方向：右、下、左、上
vector<int> printMatrix(vector<vector<int> > matrix) {
    vector<int> res;
    if(matrix.empty())
        return res;
    int width = matrix[0].size(),temp=0,flag,flag1,i,j,k,l;//temp 初始行号
    int length = matrix.size();
    while(length>=1){ //若length的为1，则遍历结束
        flag = 0;
        flag1 = 0;
        //右
        for (i = temp; i < width && temp < length; ++i) {
            res.push_back(matrix[temp][i]);
        }
        //下
        for (j = temp+1; j < length && temp < width; ++j) {
            res.push_back(matrix[j][i-1]);
            flag = 1;
        }
        //左
        for (k = width-2; (k >= temp) && flag; --k) {
            res.push_back(matrix[j-1][k]);
            flag1 = 1;
        }
        //上
        for (l = length-2; l > temp && flag1; --l) {
            res.push_back(matrix[l][k+1]);
        }
        width = width - 1;
        length = length - 1;
        temp = temp + 1;
    }
    return res;
}


//二叉搜索树与双向链表
//1.递归方法
//2.中序遍历方法（利用性质：二叉搜索树的中序遍历是一个有序的序列）
TreeNode* Convert_Rec(TreeNode* pRootOfTree, int flag) {
    if(pRootOfTree==nullptr)
        return nullptr;
    if(pRootOfTree->left){  //左子树存在
        TreeNode* max_Node = Convert_Rec(pRootOfTree->left, 1);
        pRootOfTree->left = max_Node;
        if(max_Node)
            max_Node->right = pRootOfTree;
    }
    if(pRootOfTree->right){ //右子树存在
        TreeNode* min_Node = Convert_Rec(pRootOfTree->right, 2);
        pRootOfTree->right = min_Node;
        if(min_Node)
            min_Node->left = pRootOfTree;
    }
    if(flag==1){    //返回最大节点，也就是最右边的节点
        while(pRootOfTree->right){
            pRootOfTree = pRootOfTree->right;
        }
    }else if(flag==2){  //返回最小节点
        while(pRootOfTree->left){
            pRootOfTree = pRootOfTree->left;
        }
    }
    return pRootOfTree;
}


TreeNode* Convert(TreeNode* pRootOfTree)
{
    //0表示从根节点开始
    return Convert_Rec(pRootOfTree,0);
}


//字符串的排列
//处理字符串中有重复字符的情况
//对于重复的字符，在每个位置只考虑一次

void Permutation_Rec(vector<string> & res, char seq[], map<char,int> letters, int index, int len) {
    if(index==len){
        seq[index] = '\0';
        //string temp(seq);
        res.push_back(string(seq));
        return;
    }
    for (auto i = letters.begin(); i != letters.end(); ++i) {
        if(i->second == 0){
            continue;
        } else {
            seq[index] = i->first;
            i->second = i->second - 1;
            Permutation_Rec(res,seq,letters,index+1,len);
            i->second = i->second + 1;
        }
    }

}

vector<string> Permutation(string str) {
    vector<string> res;
    if(str.empty())
        return res;
    //使用一个dict去记录str中的元素
    int len = str.length();
    map<char,int> letters;
    for (int i = 0; i < len; ++i) {
        if(letters[str[i]])
            letters[str[i]]++;
        else
            letters[str[i]] = 1;
    }
    //递归去求全排列
    char seq[len];
    Permutation_Rec(res,seq,letters,0,len);
    return res;
}

//数组中出现次数超过一半的数字
int MoreThanHalfNum_Solution(vector<int> numbers) {
    if(numbers.empty())
        return 0;
    int len = numbers.size();
    //使用map
    unordered_map<int,int> number_times;
    for (int i = 0; i < len ; ++i) {
        if(number_times[numbers[i]]){
            number_times[numbers[i]]++;
        } else
            number_times[numbers[i]] = 1;
    }
    //遍历map,找到出现次数最高的数
    int max = 0;
    int res;
    for (auto j = number_times.begin(); j != number_times.end() ; ++j) {
        if(j->second>max) {
            max = j->second;
            res = j->first;
        }
    }
    return max>(len/2) ? res : 0;

}

//找出其中最小的K个数
//采用最小堆的做法
//更新的时间复杂度o(nlogn)
vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {

}

//整数中1出现的次数
int NumberOf1Between1AndN_Solution(int n)
{
    //前100的个数
    vector<int> res = {2,12,13,14,15,16,17,18,19,21};
    vector<int> res1 = {2,10,1,1,1,1,1,1,1,2};
    int temp;
    int sum_of_one = 0; //1的个数

    if(n<=100){
        temp = n/10;
        sum_of_one += res[temp-1];
        if((temp*10+1)<=n){
            sum_of_one++;
            if(temp==1){
                sum_of_one += n-temp*10;
            }
        }
    }else{
        sum_of_one = 21;
        temp = 0;
        for (int i = 101; i < n; i = i + 10) {

            //若最高位为1，则每个循环都要加10
            int temp1 = i/100,temp2 = 0,count=0;
            while(temp1){
                temp2 = temp1%10;
                temp1 = temp1/10;
                if(temp2==1)
                    count++;
            }
            if(count>=1)
                sum_of_one += res1[temp++] + 10*count;
            else
                sum_of_one += res1[temp++];
            if(temp==10) {
                temp = 0;
                if(count>=1)
                    sum_of_one--;
            }
        }

    }

    return sum_of_one;

}

struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
            val(x), next(NULL) {
    }
};

//求两个链表的第一个公共节点
ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
    //遍历两条链表，求长度
    ListNode *pHead1_temp = pHead1, *pHead2_temp = pHead2;
    int count1=0,count2=0;
    while(pHead1_temp){
        count1++;
        pHead1_temp = pHead1_temp->next;
    }
    while(pHead2_temp){
        count2++;
        pHead2_temp = pHead2_temp->next;
    }
    //长的先遍历
    int diff = count1-count2;
    if (diff>0){
        while(diff){
            pHead1 = pHead1->next;
            diff--;
        }
    }
    else{
        while(diff){
            pHead2 = pHead2->next;
            diff--;
        }
    }
    //一起遍历
    while(pHead1 && pHead1!=pHead2){
        pHead1 = pHead1->next;
        pHead2 = pHead2->next;
    }
    return pHead1;
}

int comp( const void * p, const void * q)
{
    return ( * ( int * ) p - * ( int * ) q) ;
}

//数组中只出现一次的数字
void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
    if(data.empty()){
        num1 = nullptr;
        num2 = nullptr;
    }
    //对数组进行排序
    qsort(&data[0],data.size(),sizeof(data[0]),comp);
    //查找重复的元素
    //每两个元素比较
    int flag = 0;
    for (int j = 0; j < data.size(); ++j) {
        if(data[j]==data[j+1])
            j++;
        else{
            if(flag == 0){
                *num1 = data[j];
                flag = 1;
            }
            else{
                *num2 = data[j];
                break;
            }
        }
    }

}

//根据位运算来做
//思路：将数组中所有数进行异或，则结果为两个不同的数字的异或结果。再从这结果中选取第一个出现1的位数，
//接着把原数组分成两组，分组标准是第x位是否为1。如此，相同的数字肯定在同一组，不同数字在不同组。
//再对每组做异或计算，则出来的结果就是两个不同的数字

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};

//把二叉树打印成多行
vector<vector<int> > Print(TreeNode* pRoot) {
    //使用队列进行
    queue<TreeNode*> temp;
    vector<vector<int>> res;
    if(pRoot== nullptr)
        return res;
    temp.push(pRoot);
    int layer_num = 1;
    int layer_next_num = 0;
    while(!temp.empty()){
        vector<int> layer_temp;
        while(layer_num) {
            TreeNode* node_temp = temp.front();
            if (node_temp->left) {
                temp.push(node_temp->left);
                layer_next_num++;
            }
            if (node_temp->right) {
                temp.push(node_temp->right);
                layer_next_num++;
            }
            layer_num--;
            layer_temp.push_back(node_temp->val);
            temp.pop();
        }
        layer_num = layer_next_num;
        res.push_back(layer_temp);
        layer_next_num = 0;
    }
    return res;
}

//路径的回溯
bool back_tracking_path(char* matrix, int* flag, int rows, int cols, char* str, int pos) {
    if(str[0]=='\0')
        return true;
    int row_t,col_t;    //当前位置pos的行号和列号
    row_t = pos / cols;
    col_t = pos % cols;
    bool res = false;
    int temp = -1;
    //四个方向
    if(row_t-1 >=0){
        temp = (row_t-1)*cols + col_t;
        if(flag[temp]&&matrix[temp] == str[0]){
            flag[temp] = 0;
            res = back_tracking_path(matrix,flag,rows,cols,str+1,temp);
        }else temp = -1;
    }
    if(!res&&col_t-1>=0){
        if(temp!=-1)
            flag[temp] = 1;
        temp = row_t*cols + col_t-1;
        if(flag[temp]&&matrix[temp] == str[0]){
            flag[temp] = 0;
            res = back_tracking_path(matrix,flag,rows,cols,str+1,temp);
        }else temp = -1;
    }

    if(!res&&row_t+1<rows){
        if(temp!=-1)
            flag[temp] = 1;
        temp = (row_t+1)*cols + col_t;
        if(flag[temp]&&matrix[temp] == str[0]){
            flag[temp] = 0;
            res = back_tracking_path(matrix,flag,rows,cols,str+1,temp);
        }else temp = -1;
    }

    if(!res&&col_t+1<cols){
        if(temp!=-1)
            flag[temp] = 1;
        temp = row_t*cols + col_t+1;
        if(flag[temp]&&matrix[temp] == str[0]){
            flag[temp] = 0;
            res = back_tracking_path(matrix,flag,rows,cols,str+1,temp);
        }
    }

    return res;

}

//矩阵中的路径
//row和cols指定了矩阵的行和列数
bool hasPath(char* matrix, int rows, int cols, char* str)
{
    if(matrix == nullptr || str == nullptr)
        return false;
    //定义一个二维数组，标记矩阵中的格子是否被遍历过
    int len = rows*cols;
    int flag[len];
    for (int j = 0; j < len; ++j) {
        flag[j] = 1;
    }
    bool res;

    //先遍历找到矩阵中与str[0]相同的元素
    for (int i = 0; i < len; ++i) {
        if(matrix[i]==str[0]){

            flag[i] = 0;
            res = back_tracking_path(matrix,flag,rows,cols,str+1,i);
            if(res)
                return true;
            flag[i] = 1;
        }
    }
    return false;
}

#endif //LEETCODE_QUESTION_H
