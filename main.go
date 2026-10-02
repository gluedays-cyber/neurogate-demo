package main

import (
	"context"
	"fmt"
	"log"

	"intellibranch"
)

// 1. 비즈니스 액션 핸들러 정의
func handleRefund(ctx context.Context, payload any) error {
	fmt.Printf("[ACTION: Refund]   환불 처리 진행: '%v'\n", payload)
	return nil
}

func handleDelivery(ctx context.Context, payload any) error {
	fmt.Printf("[ACTION: Delivery] 배송 상태 추적 진행: '%v'\n", payload)
	return nil
}

func handleAccount(ctx context.Context, payload any) error {
	fmt.Printf("[ACTION: Account]  계정 보안 검증 진행: '%v'\n", payload)
	return nil
}

func handleFallback(ctx context.Context, payload any) error {
	fmt.Printf("[FALLBACK: Safety] 저신뢰도 요청 안전 격리: '%v'\n", payload)
	return nil
}

func main() {
	modelPath := "weights/intent.bin"
	dataPath := "data/sample_dataset.csv"

	// 1. 단일 라이브러리 API 호출로 가중치 자동 훈련(또는 로드) 및 라우터 인스턴스 획득
	router, err := intellibranch.OpenOrTrain(dataPath, modelPath, 0.60)
	if err != nil {
		log.Fatalf("IntelliBranch 라우터 초기화 실패: %v", err)
	}

	// 2. 지능형 분기(Branch) 및 Fallback 바인딩
	router.
		Branch("Refund", handleRefund).
		Branch("Delivery", handleDelivery).
		Branch("Account", handleAccount).
		Fallback(handleFallback)

	// 3. 지능형 분기 디스패치 실행
	testQueries := []string{
		"I want to cancel my payment and request a refund",
		"When will my delivery package arrive",
		"Forgot my account password",
		"Please refund my purchase",
		"Track my shipment status",
		"Completely random gibberish noise 12345!@#$",
		"hey where is my stuff it was supposed to get here yesterday",
		"can u cancel order #49281? i bought it by mistake",
		"bruh the reset link is not sending to my email, fix this",
		"got charged twice on my card, refund the extra charge asap",
		"item arrived totally smashed, want my money back",
		"cant log into my acct keeps saying wrong password",
		"tracking says delivered but nothing is in my mailbox",
		"yo i typed the wrong apt number, can someone update the address before it ships",
		"sent the return box a week ago, when do i get my refund?",
	}

	fmt.Println("=== IntelliBranch 독립 데모 가동 ===")
	ctx := context.Background()
	for _, query := range testQueries {
		if err := router.Dispatch(ctx, query, query); err != nil {
			log.Printf("Dispatch 오류: %v", err)
		}
	}
	fmt.Println("=== 모든 질의가 마이크로초 단위로 지능형 분기 완료됨 ===")
}
