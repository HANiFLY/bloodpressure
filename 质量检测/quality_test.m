function [result_flag, ptt, heart_rate_result] = quality_test(heart_rate, ECG)

fs = 50;  %采样频率50Hz
% NH = length(heart_rate);  % 脉搏波样本数量
FFT_start = 1;
FFT_length = 500; % 脉搏波样本数量
FFT_end = FFT_start + FFT_length;
ptt = 0; % 数据初始化
heart_rate_result = 0; % 数据初始化

while 1
    
    % 若运行到这一步，说明已经走完了全部数据，但始终没有找到合格的信号
    if FFT_end >= length(heart_rate)
        result_flag = -2;
        return;
    end

    % 去除噪声点，若噪声点的阈值低于20000，则重置为后一个点的值
%     for i = FFT_start:FFT_end
%         if heart_rate(1,i)<20000
%             heart_rate(1,i) = heart_rate(1,i+1);
%         end
%     end 

    FFT_heart_rate_X = heart_rate(:,FFT_start:FFT_end);
    FFT_heart_rate_Y = fft(FFT_heart_rate_X);
    FFT_heart_rate_Y = FFT_heart_rate_Y(2:end);  %第一个元素是所有元素的和
    FFT_heart_rate_Y = abs(FFT_heart_rate_Y); 
    f = (1:FFT_length/2)*fs/FFT_length;  %fft频率轴归一化
%     subplot(2,2,2);
%     plot(f,FFT_heart_rate_Y(FFT_start:FFT_end/2));
    figure
    plot(f,FFT_heart_rate_Y(FFT_start:FFT_end/2));
    title('FFT heart rate');
    [mp,index] = max(FFT_heart_rate_Y);  
    f_heart_rate = f(index);
    while (f_heart_rate >= 2)||(f_heart_rate <= 0.5)
        
        if FFT_end >= length(heart_rate)-500
            result_flag = -1;
%             window_heart_rate = window_heart_rate_OLD;
%             ptt = abs(calculate_ptt(window_heart_rate,window_ECG));
            return;
        end
        
        FFT_start = FFT_end + 1;
        FFT_end = FFT_start + FFT_length;
        FFT_heart_rate_X = heart_rate(:,FFT_start:FFT_end);
        FFT_heart_rate_Y = fft(FFT_heart_rate_X);
        FFT_heart_rate_Y = FFT_heart_rate_Y(2:end);  %第一个元素是所有元素的和
        FFT_heart_rate_Y = abs(FFT_heart_rate_Y); 
        f = (1:FFT_length/2)*fs/FFT_length;  %fft频率轴归一化
        [mp,index] = max(FFT_heart_rate_Y);  
        f_heart_rate = f(index);
    end

    heart_rate_result = f_heart_rate * 60;
    window_length_fft = 50 / f_heart_rate;
    window_length_OLD = round(window_length_fft);

    % 微调窗口大小，先扩大，再缩小，找到最合适的窗口大小
    while 1
        [R_heart_rate_OLD,window_heart_rate_OLD] = calculate_corrcoef(FFT_heart_rate_X, window_length_OLD);
        result_heart_rate_OLD = mean(R_heart_rate_OLD);

        window_length_NEW = window_length_OLD + 1;

        [R_heart_rate_NEW,window_heart_rate_NEW] = calculate_corrcoef(FFT_heart_rate_X, window_length_NEW);
        result_heart_rate_NEW = mean(R_heart_rate_NEW);

        if result_heart_rate_NEW > result_heart_rate_OLD
            window_length_OLD = window_length_NEW;
        else
            window_length_OLD = window_length_OLD;
            break;
        end
    end

    window_length_OLD = round(window_length_fft);

    while 1
        [R_heart_rate_OLD,window_heart_rate_OLD] = calculate_corrcoef(FFT_heart_rate_X, window_length_OLD);
        result_heart_rate_OLD = mean(R_heart_rate_OLD);

        window_length_NEW = window_length_OLD - 1;

        [R_heart_rate_NEW,window_heart_rate_NEW] = calculate_corrcoef(FFT_heart_rate_X, window_length_NEW);
        result_heart_rate_NEW = mean(R_heart_rate_NEW);

        if result_heart_rate_NEW > result_heart_rate_OLD
            window_length_OLD = window_length_NEW;
            result_heart_rate = result_heart_rate_NEW;
        else
            window_length_OLD = window_length_OLD;
            result_heart_rate = result_heart_rate_OLD;
            break;
        end
    end

    window_length = round(window_length_OLD);
 
    % 如果脉搏波信号分窗的相似程度没有达到0.9，则要进行挪窗，挪动的长度为目前FFT结果下一个波/窗口的长度
    if result_heart_rate < 0.9
        FFT_start = FFT_start + window_length;
        FFT_end = FFT_start + FFT_length;
        disp('hr:');
        disp(result_heart_rate);
    else  % 脉搏波信号精度达标，则要检测心电信号是否达标
        disp('hr:');
        disp(result_heart_rate);
        FFT_ECG_X = ECG(:,FFT_start:FFT_end);
        [R_ECG,window_ECG] = calculate_corrcoef(FFT_ECG_X, window_length);
        result_ECG = mean(R_ECG);
        if result_ECG < 0.69 % 心电信号不达标，挪动脉搏波窗口
            FFT_start = FFT_start + window_length;
            FFT_end = FFT_start + FFT_length;
            disp('ECG:');
            disp(result_ECG);
        else % 心电信号达标，计算ptt
            disp('ECG:');
            disp(result_ECG);
            window_heart_rate = window_heart_rate_OLD;
            ptt = abs(calculate_ptt(window_heart_rate,window_ECG));
            result_flag = 1;
%             subplot(2,1,1);
%             plot(FFT_heart_rate_X);
%             subplot(2,1,2);
%             plot(FFT_ECG_X);
            return;
        end
    end
    

end
end


