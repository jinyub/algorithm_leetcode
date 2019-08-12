//
// Created by xiping on 2019-02-17.
//

#ifndef LEETCODE_REMOVEDUPLICATES_H
#define LEETCODE_REMOVEDUPLICATES_H

#include <vector>

using namespace std;

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        //int len = nums.size();
        //i表示遍历整个数组的索引
        if (nums.empty())
            return 0;
        int temp = nums[0];
        int j = 1; //j表示存储的索引
        int count = 1;
        for (unsigned int i = 1; i < nums.size(); ++i) {
            if(nums[i]>temp){
                temp = nums[i];
                nums[j] = nums[i];
                j++;
                count++;
            }
        }

        return count;
    }
};

#endif //LEETCODE_REMOVEDUPLICATES_H
