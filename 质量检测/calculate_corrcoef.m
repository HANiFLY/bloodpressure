% 分窗并计算相关系数的函数
function [R,window] = calculate_corrcoef(A, window_length)

window_start = 1;
window_end = window_start + window_length - 1;
i = 1;

while i < 6
    window(i,1:window_length) = A(:,window_start:window_end);
    window_start = window_end + 1;
    window_end = window_end + window_length;
    i = i+1;
end

    R12 = corrcoef(window(1,:), window(2,:));
    R13 = corrcoef(window(1,:), window(3,:));
    R14 = corrcoef(window(1,:), window(4,:));
    R15 = corrcoef(window(1,:), window(5,:));

    r12 = mean(mean(R12)); 
    r13 = mean(mean(R13));
    r14 = mean(mean(R14));
    r15 = mean(mean(R15));
    
    R = [r12, r13, r14, r15];

end