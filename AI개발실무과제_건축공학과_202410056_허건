import pandas as pd
import numpy as np
from typing import Tuple, List

class LevenshteinChatbot:
    """
    레벤슈타인 거리를 기반으로 한 챗봇 클래스
    사용자의 질문과 가장 유사한 학습 데이터의 질문을 찾아 해당 답변을 반환합니다.
    """
    
    def __init__(self, csv_path: str):
        """
        챗봇 초기화 함수
        
        Args:
            csv_path (str): 학습 데이터가 포함된 CSV 파일 경로
        """
        # CSV 파일 읽기
        self.df = pd.read_csv(csv_path)
        
        # 질문과 답변 리스트 추출
        self.questions = self.df['Q'].tolist()
        self.answers = self.df['A'].tolist()
        
        print(f"학습 데이터 로딩 완료: {len(self.questions)}개의 질문-답변 쌍")
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        두 문자열 간의 레벤슈타인 거리를 계산합니다.
        
        레벤슈타인 거리는 한 문자열을 다른 문자열로 변환하는데 필요한
        최소 편집 횟수(삽입, 삭제, 치환)를 의미합니다.
        
        Args:
            s1 (str): 첫 번째 문자열
            s2 (str): 두 번째 문자열
            
        Returns:
            int: 두 문자열 간의 레벤슈타인 거리
        """
        # 두 문자열의 길이
        m, n = len(s1), len(s2)
        
        # DP 테이블 초기화
        # dp[i][j]는 s1[:i]와 s2[:j] 사이의 레벤슈타인 거리
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 베이스 케이스: 빈 문자열과의 거리는 문자열 길이와 같음
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # DP 테이블 채우기
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 문자가 같은 경우: 편집 불필요
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # 문자가 다른 경우: 삽입, 삭제, 치환 중 최소값 + 1
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # 삭제
                        dp[i][j-1],    # 삽입
                        dp[i-1][j-1]   # 치환
                    )
        
        return dp[m][n]
    
    def normalized_levenshtein_distance(self, s1: str, s2: str) -> float:
        """
        정규화된 레벤슈타인 거리를 계산합니다.
        
        정규화를 통해 문자열 길이에 관계없이 0~1 사이의 값으로 변환합니다.
        
        Args:
            s1 (str): 첫 번째 문자열
            s2 (str): 두 번째 문자열
            
        Returns:
            float: 정규화된 레벤슈타인 거리 (0: 완전 일치, 1: 완전 불일치)
        """
        distance = self.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        # 빈 문자열 처리
        if max_len == 0:
            return 0.0
        
        return distance / max_len
    
    def find_most_similar_question(self, user_question: str) -> Tuple[int, float]:
        """
        사용자 질문과 가장 유사한 학습 데이터의 질문을 찾습니다.
        
        Args:
            user_question (str): 사용자가 입력한 질문
            
        Returns:
            Tuple[int, float]: (가장 유사한 질문의 인덱스, 레벤슈타인 거리)
        """
        # 모든 질문과의 레벤슈타인 거리 계산
        distances = []
        for i, question in enumerate(self.questions):
            # 정규화된 레벤슈타인 거리 사용
            distance = self.normalized_levenshtein_distance(
                user_question.lower(), 
                question.lower()
            )
            distances.append(distance)
        
        # 가장 작은 거리를 가진 질문의 인덱스 찾기
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        return min_distance_idx, min_distance
    
    def get_response(self, user_question: str, threshold: float = 0.7) -> str:
        """
        사용자 질문에 대한 답변을 반환합니다.
        
        Args:
            user_question (str): 사용자가 입력한 질문
            threshold (float): 유사도 임계값 (기본값: 0.7)
                              이 값보다 거리가 크면 적절한 답변을 찾지 못한 것으로 판단
            
        Returns:
            str: 챗봇의 답변
        """
        # 가장 유사한 질문 찾기
        best_idx, distance = self.find_most_similar_question(user_question)
        
        # 유사도가 임계값보다 낮은 경우 기본 메시지 반환
        if distance > threshold:
            return "죄송합니다. 적절한 답변을 찾을 수 없습니다. 다른 질문을 해주세요."
        
        # 찾은 질문과 답변 정보
        similar_question = self.questions[best_idx]
        answer = self.answers[best_idx]
        
        # 디버깅 정보 출력 (선택사항)
       # print(f"\n[디버그 정보]")
        #print(f"입력 질문: {user_question}")
        #print(f"가장 유사한 질문: {similar_question}")
        #print(f"레벤슈타인 거리: {distance:.4f}")
        #print(f"선택된 답변: {answer}")
        
        return answer
    
    def get_top_k_responses(self, user_question: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """
        사용자 질문과 가장 유사한 상위 k개의 질문-답변 쌍을 반환합니다.
        
        Args:
            user_question (str): 사용자가 입력한 질문
            k (int): 반환할 상위 결과 개수
            
        Returns:
            List[Tuple[str, str, float]]: [(질문, 답변, 거리), ...] 형태의 리스트
        """
        # 모든 질문과의 레벤슈타인 거리 계산
        distances = []
        for i, question in enumerate(self.questions):
            distance = self.normalized_levenshtein_distance(
                user_question.lower(), 
                question.lower()
            )
            distances.append((i, distance))
        
        # 거리순으로 정렬
        distances.sort(key=lambda x: x[1])
        
        # 상위 k개 결과 추출
        results = []
        for i in range(min(k, len(distances))):
            idx, distance = distances[i]
            results.append((
                self.questions[idx],
                self.answers[idx],
                distance
            ))
        
        return results


# 사용 예시
if __name__ == "__main__":
    # 챗봇 초기화
    chatbot = LevenshteinChatbot('ChatbotData.csv')
    
    # 대화형 챗봇 실행
    print("\n레벤슈타인 거리 기반 챗봇을 시작합니다.")
    print("종료하려면 'quit', 'exit', 또는 '종료'를 입력하세요.\n")
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n사용자: ").strip()
        
        # 종료 조건 확인
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("챗봇을 종료합니다. 감사합니다!")
            break
        
        # 빈 입력 처리
        if not user_input:
            print("챗봇: 질문을 입력해주세요.")
            continue
        
        # 답변 생성
        response = chatbot.get_response(user_input)
        print(f"\n챗봇: {response}")
        
        # 상위 3개 유사 질문 표시 (선택사항)
        #print("\n[참고: 유사한 질문들]")
        #top_k = chatbot.get_top_k_responses(user_input, k=3)
        #for i, (q, a, d) in enumerate(top_k, 1):
         #   print(f"{i}. 질문: {q}")
          #  print(f"   답변: {a}")
           # print(f"   거리: {d:.4f}")