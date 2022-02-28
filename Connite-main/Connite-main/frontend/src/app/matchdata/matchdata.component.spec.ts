import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MatchdataComponent } from './matchdata.component';

describe('MatchdataComponent', () => {
  let component: MatchdataComponent;
  let fixture: ComponentFixture<MatchdataComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ MatchdataComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(MatchdataComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
