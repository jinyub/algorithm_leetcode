#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
using namespace std;

#include <algorithm>


class Solution {
public:
    vector<int> data_stream;
    void Insert(int num)
    {
        data_stream.push_back(num);
    }

    double GetMedian()
    {
        //对vector进行排序
        sort(data_stream.begin(),data_stream.end());
        int len = data_stream.size();
        int mid = len/2;
        return len%2==0 ? (data_stream[mid]+data_stream[mid-1])/2 : data_stream[mid];

    }

};

//构建最大堆
priority_queue<int,vector<int>> max_heap;
//构建最小堆
priority_queue<int,vector<int>,greater<>> min_heap;
//数据流中的中位数
//采用一个最小堆和一个最大堆
void Insert(int num)
{

    if(max_heap.empty())
        max_heap.push(num);
    else if(min_heap.empty()){
        if(num>=max_heap.top()){
            min_heap.push(num);
        }else{
            int temp = max_heap.top();
            max_heap.pop();
            max_heap.push(num);
            min_heap.push(temp);
        }
    }else{
    //新插入的数字分别于最大堆和最小堆的堆顶元素进行比较，确定插入位置
    if(num<=min_heap.top()){
        if(max_heap.size()-min_heap.size()==-1||max_heap.size()==min_heap.size()){
            max_heap.push(num);
        }else if(max_heap.size()-min_heap.size()==1){
            max_heap.push(num);
            int temp = max_heap.top();
            max_heap.pop();
            min_heap.push(temp);
        }
    } else {
        if(max_heap.size()-min_heap.size()==1||max_heap.size()==min_heap.size()){
            min_heap.push(num);
        }else if(min_heap.size()-max_heap.size()==1){
            min_heap.push(num);
            int temp = min_heap.top();
            min_heap.pop();
            max_heap.push(temp);
        }
    }
    }
}

double GetMedian()
{
    int len = min_heap.size() + max_heap.size();
    bool even = len%2==0;
    if(even)
        return (min_heap.top()+max_heap.top())/2.0;
    else{
        if(min_heap.size()>max_heap.size()){
            return min_heap.top();
        }else
            return max_heap.top();

    }
}

/**
 * 不用加减乘除做加法
 * 1.使用与操作计算进位值
 * 2.使用异或操作计算相加值
 * 3.若进位值为0，则直接返回异或结果
 * @param num1
 * @param num2
 * @return
 */
int Add(int num1, int num2)
{
        int carry = (num1 & num2)<<1;
        int additive = num1 ^ num2,temp;
        while(carry){
            temp = additive;
            additive = carry ^ additive;
            carry = (carry & temp) << 1;
        }
        return additive;
}


struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};

/**
 * 按之字形打印
 * @param pRoot
 * @return
 */
vector<vector<int> > Print(TreeNode* pRoot) {
    vector<vector<int>> res;
    if(pRoot==nullptr)
        return res;
    stack<TreeNode*> zhi_s;
    queue<TreeNode*> zhi_q;
    int num = 1,num_next_level=0;    //每层数目
    int order = 1; //从左往右还是从右往左
    zhi_q.push(pRoot);
    vector<int> level = vector<int>();
    while(!zhi_q.empty()){
        TreeNode* temp = zhi_q.front();
        zhi_q.pop();
        if(temp->left) {
            num_next_level++;
            zhi_q.push(temp->left);
            if(order)
                zhi_s.push(temp->left);
        }
        if(temp->right) {
            num_next_level++;
            zhi_q.push(temp->right);
            if(order)
                zhi_s.push(temp->right);
        }
        if(order)
            level.push_back(temp->val);
        num--;
        if(num==0){
            if(order) {
                res.push_back(level);
            }
            else if(order==0){
                while(!zhi_s.empty()){
                    level.push_back(zhi_s.top()->val);
                    zhi_s.pop();
                }
                res.push_back(level);
            }
            order = order==1 ? 0 : 1;
            level = vector<int>();
            num = num_next_level;
            num_next_level = 0;
        }

    }
    return res;

}

/**
 * 序列化和反序列化二叉树
 * @param root
 * @return
 */
char* Serialize(TreeNode *root) {
    char* res = new char();
    //根据前序遍历对二叉树进行序列化
    if(root== nullptr){

    }


}
TreeNode* Deserialize(char *str) {

}


int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    vector<vector<int>> res = Print(root);

    cout << Add(5,7) << endl;

    return 0;
}