#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <stdlib.h>
#include <limits.h>
using namespace std;

#include <algorithm>
#include <string.h>

//找滑动窗口中的最大值
//这种思路相当于求固定size中的最大值，时间复杂度为O(n*size)
vector<int> maxInWindows(const vector<int>& num, unsigned int size)
{
    vector<int> res;
    if(num.empty()||size==0)
        return res;
    int max=INT_MIN,start=1,max_index=0;
    for (int j = 0; j < size; ++j) {
        if(num[j]>max){
            max = num[j];
            max_index = j;
        }
    }
    res.push_back(max);
    for (int i = start; i < num.size()-size+1; ++i) {
        if(max_index!=i-1){
            if(num[i+size-1]>max){
                max = num[i+size-1];
                max_index = i+size-1;
            }
            res.push_back(max);
        }else{
            max = INT_MIN;
            for (int j = i; j < i+size-1; ++j) {
                if(num[j]>max){
                    max = num[j];
                    max_index = j;
                }
            }
            if(num[i+size-1]>max){
                max = num[i+size-1];
                max_index = i+size-1;
            }
            res.push_back(max);
        }
    }
    return res;
}

//求滑动窗口中的最大值，使用双端队列
vector<int> maxIntWindow(const vector<int>& num, unsigned int size) {
    deque<int> hd;//滑动窗口最大为size的大小,双端队列中存的是索引值
    vector<int> res;
    if(num.empty()||size==0||size>num.size())
        return res;
    hd.push_back(0);
    for (int i = 1; i < size-1; ++i) {
        if(num[i]<num[hd.back()]){
            hd.push_back(i);
        }else{
            while(!hd.empty()&&num[i]>=num[hd.back()]){
                hd.pop_back();
            }
            hd.push_back(i);
        }
    }
    for (int j = size-1; j < num.size(); ++j) {
        if(num[j]<num[hd.back()]){
            hd.push_back(j);
            if(hd.front()<j-size+1){
                hd.pop_front();
            }
            res.push_back(num[hd.front()]);
        }else{
            while(!hd.empty()&&num[j]>=num[hd.back()]){
                hd.pop_back();
            }
            hd.push_back(j);
            res.push_back(num[hd.front()]);
        }
    }
    return res;
}

/**
 * leetcode 56
 * 合并区间
 * 先排序，再判断
 * @param intervals
 * @return
 */

bool greater_first(const vector<int> & m1, const vector<int> & m2) {
    return m1[0] < m2[0];
}

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    vector<vector<int>> res;
    if(intervals.empty())
        return res;
    //先对区间进行排序，使用STL自带的模板库
    sort(intervals.begin(),intervals.end(),greater_first);
    //再进行判断操作
    vector<int> temp={intervals[0][0],intervals[0][1]};
    for (int i = 1; i < intervals.size(); ++i) {
        if(intervals[i][0]>=temp[0]&&intervals[i][0]<=temp[1]){
            temp[0] = temp[0];
            temp[1] = temp[1]>intervals[i][1]?temp[1]:intervals[i][1];
        }else{
            res.push_back(temp);
            temp[0] = intervals[i][0];
            temp[1] = intervals[i][1];
        }
    }
    res.push_back(temp);
    return res;
}

/**
 * leetcode 75
 * 颜色分类
 * 使用常数空间的一趟排序算法
 * @param nums
 */
void sortColors(vector<int>& nums) {
    int num_one = 0;//1的个数
    if(nums.empty())
        return;
    int i=0, j=nums.size()-1,k=0;//i指向开头，j指向结尾
    while(nums[j]==2){
        j--;
    }
    for (k = 0; k <= j; ++k) {
        if(nums[k]==0){
            nums[i++] = nums[k];
        }else if(nums[k]==1){
            num_one++;
        }else{
            swap(nums[k],nums[j--]);
            if(nums[k]==0){
                nums[i++] = nums[k];
            }else if(nums[k]==1)
                num_one++;
            else {
                continue;
            }
        }
    }
    for (int l = i; l < i+num_one; ++l) {
        nums[l] = 1;
    }
}

void sortColors_r(vector<int>& nums) {
    if(nums.empty())
        return;
    int i=0,j=nums.size()-1;
    for (int curr = 0; curr <= j; ++curr) {
        if(nums[curr]==0){
            swap(nums[i++],nums[curr]);
        }else if(nums[curr]==2){
            swap(nums[curr--],nums[j--]);
        }
    }
}
/**
 * leetcode 5 最长回文子串
 * 从中间向两边扩散
 * 还有是无中心点的情况
 * @param s
 * @return
 */
string longestPalindrome(string s) {
    if(s.empty())
        return "";
    int len = s.length(),max_len=0;
    int max = 0;
    //判断全部相同的情况
    int start=0,end=0;
    for (int i = 0; i < len; ++i) {
        //有中心点情况
        int j=1,len1=0,len2=0;
        while((i-j)>=0&&(i+j)<=len&&s[i-j]==s[i+j]){
            j++;
            len1++;
        }
        len1 = 2*len1 + 1;

        //无中心点情况
        j=1;
        while((i-j)>=0&&(i+j-1)<=len&&s[i-j]==s[i+j-1]){
            j++;
            len2++;
        }
        len2 = 2*len2;

        len1 = len1 > len2 ? len1 : len2;
        if(len1>max){
            max = len1;
            start = i-len1/2;
            end = i+(len1-1)/2;
            max_len = len1;
        }
    }
    return s.substr(start,max_len);
}

/**
 * leetcode 137
 * 只出现一次的数字
 * @param nums
 * @return
 */
int singleNumber(vector<int>& nums) {

}

int main() {
//    string a;
//    cout << a.length() << endl;
    cout << longestPalindrome("aaaabaaaaa") << endl;
    return 0;
}