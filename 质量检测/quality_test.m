function [result_flag, ptt, heart_rate_result] = quality_test(heart_rate, ECG)

fs = 50;  %����Ƶ��50Hz
% NH = length(heart_rate);  % ��������������
FFT_start = 1;
FFT_length = 500; % ��������������
FFT_end = FFT_start + FFT_length;
ptt = 0; % ���ݳ�ʼ��
heart_rate_result = 0; % ���ݳ�ʼ��

while 1
    
    % �����е���һ����˵���Ѿ�������ȫ�����ݣ���ʼ��û���ҵ��ϸ���ź�
    if FFT_end >= length(heart_rate)
        result_flag = -2;
        return;
    end

    % ȥ�������㣬�����������ֵ����20000��������Ϊ��һ�����ֵ
%     for i = FFT_start:FFT_end
%         if heart_rate(1,i)<20000
%             heart_rate(1,i) = heart_rate(1,i+1);
%         end
%     end 

    FFT_heart_rate_X = heart_rate(:,FFT_start:FFT_end);
    FFT_heart_rate_Y = fft(FFT_heart_rate_X);
    FFT_heart_rate_Y = FFT_heart_rate_Y(2:end);  %��һ��Ԫ��������Ԫ�صĺ�
    FFT_heart_rate_Y = abs(FFT_heart_rate_Y); 
    f = (1:FFT_length/2)*fs/FFT_length;  %fftƵ�����һ��
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
        FFT_heart_rate_Y = FFT_heart_rate_Y(2:end);  %��һ��Ԫ��������Ԫ�صĺ�
        FFT_heart_rate_Y = abs(FFT_heart_rate_Y); 
        f = (1:FFT_length/2)*fs/FFT_length;  %fftƵ�����һ��
        [mp,index] = max(FFT_heart_rate_Y);  
        f_heart_rate = f(index);
    end

    heart_rate_result = f_heart_rate * 60;
    window_length_fft = 50 / f_heart_rate;
    window_length_OLD = round(window_length_fft);

    % ΢�����ڴ�С������������С���ҵ�����ʵĴ��ڴ�С
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
 
    % ����������źŷִ������Ƴ̶�û�дﵽ0.9����Ҫ����Ų����Ų���ĳ���ΪĿǰFFT�����һ����/���ڵĳ���
    if result_heart_rate < 0.9
        FFT_start = FFT_start + window_length;
        FFT_end = FFT_start + FFT_length;
        disp('hr:');
        disp(result_heart_rate);
    else  % �������źž��ȴ�꣬��Ҫ����ĵ��ź��Ƿ���
        disp('hr:');
        disp(result_heart_rate);
        FFT_ECG_X = ECG(:,FFT_start:FFT_end);
        [R_ECG,window_ECG] = calculate_corrcoef(FFT_ECG_X, window_length);
        result_ECG = mean(R_ECG);
        if result_ECG < 0.69 % �ĵ��źŲ���꣬Ų������������
            FFT_start = FFT_start + window_length;
            FFT_end = FFT_start + FFT_length;
            disp('ECG:');
            disp(result_ECG);
        else % �ĵ��źŴ�꣬����ptt
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


